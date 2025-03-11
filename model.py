from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import json, os
from resize_image import resize_with_padding

"""
Classes : Safety vest, NO-Safety Vest, Hardhat, 
No Hardhat, Gloves, Mask, No-Mask, Person, Safety Cone
"""

image_folder = "test/"
result_folder = "result/"
if not os.path.exists(image_folder) :
    os.makedirs(image_folder)
if not os.path.exists(result_folder) :
    os.makedirs(result_folder)
    
image_path = os.path.join(image_folder,"resized_test_image_2.jpg")
result_path = os.path.join(result_folder,"resized_test_image_2_inference.jpg")
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="AonDlnwLMihldiyHw3P9"
)
resized_img = resize_with_padding(image_path, image_path)
results = CLIENT.infer(image_path, model_id="construction-site-safety/27")
print(json.dumps(results, indent=4))
image = Image.open(image_path)
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

for result in results["predictions"] :
    x, y, width, height = result['x'], result['y'], result['width'], result['height']
    label = f"{result['class']} {result['confidence']*100}"
    x1 = x - width / 2
    y1 = y - height / 2
    x2 = x1 + width
    y2 = y1 + height
    bbox = draw.textbbox((x1, y2 - 20), label, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    draw.rectangle([x1, y2 - text_height, x1 + text_width, y2], fill="white")
    draw.text((x1, y2 - text_height), label, fill="black", font=font)
    
image.show()
image.save(result_path)

