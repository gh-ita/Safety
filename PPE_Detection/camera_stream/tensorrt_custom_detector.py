from threading import Thread, Lock 
import tensorrt as trt
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from time import sleep
import argparse
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

lock = Lock()
run_signal = False
exit_signal = False

def xywh2abcd(xywh, im_shape):
    """converts the xywh format to the abcd format for the ZED SDK"""
    # xywh: [x_center, y_center, width, height]
    # im_shape: [height, width]
    # output: [A, B, C, D] (4 points)
    # A: top-left, B: top-right, C: bottom-right, D: bottom-left
     
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) #* im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) #* im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) #* im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) #* im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max
    return output

class YoloTRT:
    def __init__(self, engine_path, conf_thresh=0.4, iou_thresh=0.45):

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        cuda.init()
        self.device = cuda.Device(0)
        self.context = self.device.make_context()
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Setup I/O bindings
        self.bindings = []
        self.input_shape = None
        self.output_shapes = []
        self.output_names = []
        self.binding_addrs = {}
        

        for binding in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(binding)
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            type = self.engine.get_tensor_mode(tensor_name)
            if type == trt.TensorIOMode.INPUT:
                # This is the input
                self.input_shape = shape
                # Allocate host and device buffers
                size = int(np.prod(shape) * np.dtype(dtype).itemsize)
                self.binding_addrs[tensor_name] = cuda.mem_alloc(size)
            else:
                # These are the outputs
                self.output_shapes.append(shape)
                self.output_names.append(tensor_name)
                # Allocate host and device buffers
                size = int(np.prod(shape) * np.dtype(dtype).itemsize)
                self.binding_addrs[tensor_name] = cuda.mem_alloc(size)
            
            self.bindings.append(int(self.binding_addrs[tensor_name]))
    
    def __del__(self):
        if hasattr(self, "cu_ctx"):
            self.cu_ctx.pop()
            self.cu_ctx.detach()

    def preprocess(self, img, input_shape):
        """Preprocess the input image for TensorRT inference"""
        # Resize and pad to maintain aspect ratio
        h, w = img.shape[:2]
        new_shape = input_shape[2:]  # Assuming NCHW format
        
        # Calculate ratio and padding
        r = min(new_shape[0] / h, new_shape[1] / w)
        new_unpad = int(round(w * r)), int(round(h * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        # Divide padding into 2 sides
        dw /= 2
        dh /= 2
        
        # Resize
        if (w, h) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # Add padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # HWC to CHW, BGR to RGB
        img = img.transpose((2, 0, 1))[::-1]
        
        # Normalize to [0,1] and convert to float32
        img = np.ascontiguousarray(img) / 255.0
        img = img.astype(np.float32)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img, r, (dw, dh)
    
    def postprocess(self, outputs, original_img_shape, conf_thresh=None, iou_thresh=None):
        """Post-process the network's output"""
        if conf_thresh is None:
            conf_thresh = self.conf_thresh
        if iou_thresh is None:
            iou_thresh = self.iou_thresh
            
        # For YOLOv8, assuming outputs is a list where the first item contains detection results
        # Format: [x, y, w, h, conf, cls1, cls2, ...]
        boxes = []
        scores = []
        class_ids = []
        
        # Process detection outputs
        for i, det in enumerate(outputs[0]):  # Assuming first output is detection
            if det[4] > conf_thresh :  # Filter by confidence
                x, y, w, h = det[0:4]
                score = det[4]
                class_id = np.argmax(det[5:])
                
                boxes.append([x, y, w, h])
                scores.append(score)
                class_ids.append(class_id)
        
        # Apply NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
        
        # Create detection objects in the format expected by the ZED SDK
        detections = []
        for i in indices:
            if isinstance(i, list):  # OpenCV 4.2 compatibility
                i = i[0]
                
            box = boxes[i]
            detection = type('ObjectDetection', (), {})()
            detection.xywh = np.array([[box[0], box[1], box[2], box[3]]])
            detection.conf = scores[i]
            detection.cls = class_ids[i]
            detections.append(detection)
            
        return detections
    
    def predict(self, img, conf_thresh=None, iou_thresh=None):
        """Perform inference on an image"""
        # Preprocess image
        input_img, ratio, (dw, dh) = self.preprocess(img, self.input_shape)
        
        # Copy input data to device
        cuda.memcpy_htod(self.binding_addrs[self.engine.get_tensor_name(0)], input_img)
        
        # Run inference
        self.context.execute_v2(self.bindings)
        
        # Get outputs
        outputs = []
        for i, output_name in enumerate(self.output_names):
            # Get output shape
            shape = self.output_shapes[i]
            print(shape)
            output = np.empty(shape, dtype=np.float32)
            cuda.memcpy_dtoh(output, self.binding_addrs[output_name])
            outputs.append(output)
        
        # Post-process
        detections = self.postprocess(outputs, img.shape, conf_thresh, iou_thresh)
        
        return detections
    
def detections_to_custom_box(detections, im0):
    """Convert detections to ZED CustomBox format"""
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
        obj.label = det.cls
        obj.probability = det.conf
        obj.is_grounded = False
        output.append(obj)
    return output

def tensorrt_thread(engine_path, img_size, conf_thres=0.4, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections

    print("Initializing TensorRT Engine...")

    model = YoloTRT(engine_path, conf_thres, iou_thres)

    while not exit_signal:
        if run_signal:
            lock.acquire()

            img = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)
            # Perform inference using TensorRT engine
            det = model.predict(img, conf_thres, iou_thres)

            # ZED CustomBox format
            detections = detections_to_custom_box(det, image_net)
            lock.release()
            run_signal = False
        sleep(0.01)

def main():
    global image_net, exit_signal, run_signal, detections

    # Start TensorRT inference thread
    capture_thread = Thread(target=tensorrt_thread, 
                          kwargs={'engine_path': opt.engine, 
                                  'img_size': opt.img_size, 
                                  'conf_thres': opt.conf_thres,
                                  'iou_thres': opt.iou_thres})
    capture_thread.start()

    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat()

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False  # designed to give person pixel mask with internal OD
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    point_cloud_render = sl.Mat()
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image_left = sl.Mat()
    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # Utilities for tracks view
    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
    # Camera pose
    cam_w_pose = sl.Pose()

    while viewer.is_available() and not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections
            lock.acquire()
            # -- Ingest detections
            zed.ingest_custom_box_objects(detections)
            lock.release()
            zed.retrieve_objects(objects, obj_runtime_param)

            # -- Display
            # Retrieve display data
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            point_cloud.copy_to(point_cloud_render)
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # 3D rendering
            viewer.updateData(point_cloud_render, objects)
            # 2D rendering
            np.copyto(image_left_ocv, image_left.get_data())
            cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
            global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            # Tracking view
            track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

            cv2.imshow("ZED | 2D View and Birds View", global_image)
            key = cv2.waitKey(10)
            if key == 27 or key == ord('q') or key == ord('Q'):
                exit_signal = True
        else:
            exit_signal = True

    viewer.exit()
    exit_signal = True
    zed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, required=True, help='path to TensorRT engine file (.engine)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file, if not passed, use the plugged camera instead')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
    opt = parser.parse_args()

    main()