import cv2
from ultralytics import YOLO
import os
import numpy as np

# Load the YOLO model (ensure you have the model downloaded)
model = YOLO("yolov8n-seg.pt")  # Use a segmentation model

# Read the image
image_path = '/home/ttyh/hot3d/hot3d/dataset/images/bread.jpg'
img = cv2.imread(image_path)

# Perform segmentation
results = model(image_path)

# Get image dimensions
img_height, img_width = img.shape[:2]
max_mask_area = (img_width * img_height) / 4  # Maximum allowed mask size (1/4 of the image)

# Loop through detected objects and process valid masks
for result in results:
    masks = result.masks.xy  # Get segmentation masks as polygons
    for mask in masks:
        # Convert the mask coordinates to a NumPy array of integers
        mask_polygon = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))

        # Calculate the mask area
        area = cv2.contourArea(mask_polygon)

        # Filter out large masks (only process smaller masks)
        if area < max_mask_area:
            # Create an overlay mask to fill the detected region
            mask_filled = np.zeros_like(img, dtype=np.uint8)
            cv2.fillPoly(mask_filled, [mask_polygon], color=(0, 255, 0))  # Green color fill

            # Add the filled mask with transparency
            img = cv2.addWeighted(img, 1, mask_filled, 0.5, 0)

# Display segmented image
detected = os.path.join('/home/ttyh/hot3d/hot3d/dataset/cropped_objects', f"segbread.jpg")
cv2.imwrite(detected, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
