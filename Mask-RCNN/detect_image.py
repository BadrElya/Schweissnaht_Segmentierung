import os
import argparse
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

def main(image_path, model_path):
    # Set up configuration and model
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir="logs")

    # Load weights
    model.load_weights(model_path, by_name=True)

    # Load and predict image
    image = skimage.io.imread(image_path)
    if image.ndim == 2:
        image = skimage.color.gray2rgb(image)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]

    results = model.detect([image])

    r = results[0]
    class_names = ['BG', 'weld']  # Background and your class 'weld'

    if r['scores'].any():
        highest_score_index = np.argmax(r['scores'])
        class_id = r['class_ids'][highest_score_index]
        score = r['scores'][highest_score_index]
        annotated_image = image.copy()

        red_color = (255, 0, 0)  # Red in BGR
        if class_id == 1:
            mask = r['masks'][:, :, highest_score_index]
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_image, contours, -1, red_color, 2)

            y1, x1, y2, x2 = r['rois'][highest_score_index]
            label = f"{class_names[class_id]} {score:.2f}"
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red_color, 2)

        plt.figure(figsize=(8, 8))
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.show()

        output_image_path = "annotated_image.jpg"
        cv2.imwrite(output_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    else:
        print("Keine Objekte in dem Bild erkannt.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection with Mask R-CNN")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    args = parser.parse_args()
    main(args.image_path, args.model_path)
