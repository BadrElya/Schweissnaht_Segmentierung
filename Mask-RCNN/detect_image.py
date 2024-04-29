import os
import numpy as np
import skimage.io
import cv2
from matplotlib import pyplot as plt

from mrcnn import model as modellib
from mrcnn.config import Config
from mrcnn import visualize

class CustomConfig(Config):
    NAME = "object"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  # Background + weld
    DETECTION_MIN_CONFIDENCE = 0.3

class InferenceConfig(CustomConfig):
    DETECTION_NMS_THRESHOLD = 0.3

# Set up configuration and model
inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir="logs")

# Load weights
model.load_weights('Deck01_train2_mrcnn.h5', by_name=True)

# Load and predict image
image_path = "1.png"
image = skimage.io.imread(image_path)
# Überprüfen, ob das Bild nur zwei Dimensionen hat (d.h., es ist ein Graustufenbild)
if image.ndim == 2:
    image = skimage.color.gray2rgb(image)
elif image.ndim == 3 and image.shape[2] == 4:
    # Konvertieren von RGBA zu RGB, falls das Bild 4 Kanäle (inklusive Alpha) hat
    image = image[..., :3]

results = model.detect([image])

# Process the highest scoring result
r = results[0]
class_names = ['BG', 'weld']  # Background and your class 'weld'

# Find the index of the highest scoring instance
if r['scores'].any():
    highest_score_index = np.argmax(r['scores'])
    class_id = r['class_ids'][highest_score_index]
    score = r['scores'][highest_score_index]
    # Create a copy of the original image to draw on
    annotated_image = image.copy()

    # Draw contours of the mask with the highest score in red color
    red_color = (255, 0, 0)  # Red in BGR
    if class_id == 1:  # Assuming 'weld' class_id is 1
        mask = r['masks'][:, :, highest_score_index]
        # Find contours of the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated_image, contours, -1, red_color, 2)  # Draw contours in red color

        # Label
        y1, x1, y2, x2 = r['rois'][highest_score_index]
        label = f"{class_names[class_id]} {score:.2f}"
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red_color, 2)

    # Display the annotated image
    plt.figure(figsize=(8, 8))
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.show()

    # Save the annotated image to a file
    output_image_path = "annotated_image.jpg"
    cv2.imwrite(output_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
else:
    print("Keine Objekte in dem Bild erkannt.")
