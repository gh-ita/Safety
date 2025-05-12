import cv2
import json
import os

# === Configuration ===
image_path = "test_images/construction_site.png"  
output_json = "danger_zones.json"

# === Globals ===
drawing = False
current_polygon = []
all_polygons = []

def click_event(event, x, y, flags, param):
    global drawing, current_polygon, all_polygons

    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_polygon) >= 3:
            all_polygons.append(current_polygon.copy())
        current_polygon = []

def draw_polygons(img):
    for poly in all_polygons:
        pts = np.array(poly, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    if len(current_polygon) >= 2:
        for i in range(len(current_polygon) - 1):
            cv2.line(img, current_polygon[i], current_polygon[i + 1], (0, 255, 0), 1)

import numpy as np

def main():
    global current_polygon, all_polygons

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    image = cv2.imread(image_path)
    cv2.namedWindow("Draw Danger Zones")
    cv2.setMouseCallback("Draw Danger Zones", click_event)

    while True:
        clone = image.copy()
        draw_polygons(clone)
        cv2.imshow("Draw Danger Zones", clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            all_polygons = []
            current_polygon = []
            print("Reset all polygons.")
        elif key == ord("s"):
            # Save polygons to JSON
            with open(output_json, "w") as f:
                json.dump(all_polygons, f)
            print(f"Saved danger zones to {output_json}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
