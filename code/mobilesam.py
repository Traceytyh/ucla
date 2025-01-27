import os
import cv2
from ultralytics import SAM, FastSAM
import numpy as np
import torch

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the model on GPU if available
model = SAM("mobile_sam.pt").to(device)

# Display model information (optional)
model.info()

# Define thresholds for filtering bounding boxes
MAX_AREA_THRESHOLD = 50000  # Maximum area of a bounding box to consider
MIN_AREA_THRESHOLD = 1200   # Minimum area of a bounding box to consider
MAX_ASPECT_RATIO = 2.0     # Maximum aspect ratio to consider a segment elongated
MIN_WHITE_INTENSITY = 125  # Minimum mean intensity to consider a blob "white"

# Path to the images directory
image_folder = '/home/ttyh/hot3d/hot3d/dataset/images'
output_folder = '/home/ttyh/hot3d/hot3d/dataset/cropped_objects'

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Process each image in the folder
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)

    # Run inference on the current frame to get bounding boxes
    results = model(source=image_path, stream=True)

    for r in results:
        masks = r.masks  # Get segmentation masks

        # Extract bounding boxes from masks
        for mask in masks.xy:
            # Convert the list of coordinates to a NumPy array (polygon format)
            mask_polygon = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))

            # Fit a bounding rectangle to the mask and calculate aspect ratio and area
            x, y, w, h = cv2.boundingRect(mask_polygon)
            area = w * h  # Calculate bounding box area

            # Filter out unwanted bounding boxes based on area and aspect ratio
            if y < 75 or h == 0 or w == 0:
                continue
            aspect_ratio = max(w / h, h / w)

            if area > MAX_AREA_THRESHOLD or area < MIN_AREA_THRESHOLD:
                continue  # Skip this bounding box if it's too large or too small

            # Crop the detected object from the original image
            cropped_object = image[y:y+h, x:x+w]
            
            # Save the cropped object
            #cropped_filename = os.path.join(output_folder, f"cropped_{image_name}_{x}_{y}.jpg")
            #cv2.imwrite(cropped_filename, cropped_object)
            #print(f"Saved cropped object: {cropped_filename}")

            # Draw the bounding box on the original image (green color)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image with bounding boxes
        #cv2.imshow("Detected Objects", image)
        detected = os.path.join(output_folder, f"mobilesam_{image_name}.jpg")
        cv2.imwrite(detected, image)
            
        # Break if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

# Now, you can process the cropped images with YOLO
