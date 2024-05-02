import os
import sys
import time
from glob import glob
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, precision_recall_curve, auc


def predict_and_convert_mask(image_path, model):
    start_time = time.time()
    results = model.predict(source=image_path, show=False, save=False, show_labels=False, show_conf=False, conf=0.1, save_txt=False, save_crop=False)
    end_time = time.time()
    prediction_time = end_time - start_time
    result0 = results[0]
    
    if len(results[0].boxes.conf.tolist()) > 0:
        # Create a PIL image for mask visualization
        mask_pil = Image.new('1', (result0.orig_img.shape[1], result0.orig_img.shape[0]), 0)
        draw = ImageDraw.Draw(mask_pil)
        for segment in result0.masks.xy:
            segment_list = [(int(x), int(y)) for x, y in segment]
            draw.polygon(segment_list, outline=1, fill=1)
        
        return np.array(mask_pil, dtype=np.bool_), prediction_time
    else:
        mask_pil = Image.new('1', (result0.orig_img.shape[1], result0.orig_img.shape[0]), 0)
        binary_mask = np.zeros_like(mask_pil)
        return np.array(mask_pil, dtype=np.bool_), prediction_time

def evaluate_model(model, test_data_folder, true_mask_paths, output_folder):
    # Dummy-Bild erstellen
    dummy_image = Image.new('RGB', (640, 480), color='white')
    # Dummy-Bild vorhersagen
    _ = model.predict(dummy_image)
    
    os.makedirs(output_folder, exist_ok=True)
    scores = []
    prediction_times = []

    for true_path in tqdm(true_mask_paths):
        name = os.path.basename(true_path).replace("_mask.png", ".png")
        image_path = os.path.join(test_data_folder, name)
        true_mask_path = true_path
        
        original_image = Image.open(image_path).convert("RGBA")  # Ensure image is in RGBA format
        predicted_mask, prediction_time = predict_and_convert_mask(image_path, model)
        prediction_times.append(prediction_time)

        true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)
        true_mask = (true_mask / 255.0) > 0.5  # Convert to binary mask

        # Convert binary masks to mask images
        predicted_mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8)).convert("L")
        true_mask_image = Image.fromarray((true_mask * 255).astype(np.uint8)).convert("L")

        # Create transparent overlays for masks
        red_overlay = Image.new("RGBA", original_image.size, (255, 0, 0, 0))  # Red overlay
        green_overlay = Image.new("RGBA", original_image.size, (0, 255, 0, 0))  # Green overlay

        # Draw masks onto the overlays
        red_overlay.paste((255, 0, 0, 125), (0, 0), true_mask_image)
        green_overlay.paste((0, 255, 0, 125), (0, 0), predicted_mask_image)

        # Composite overlays with the original image
        combined_image = Image.alpha_composite(original_image, red_overlay)
        combined_image = Image.alpha_composite(combined_image, green_overlay)

        # Save the final image
        combined_path = os.path.join(output_folder, name)
        combined_image.save(combined_path)

        # Flatten masks for evaluation metrics
        predicted_mask = np.array(predicted_mask, dtype=np.bool_).astype(int).flatten()
        true_mask = np.array(true_mask, dtype=np.bool_).astype(int).flatten()

        # Evaluation metrics
        acc_value = accuracy_score(true_mask, predicted_mask)
        f1_value = f1_score(true_mask, predicted_mask, average="binary")
        jac_value = jaccard_score(true_mask, predicted_mask, average="binary")
        recall_value = recall_score(true_mask, predicted_mask, average="binary")
        precision_value = precision_score(true_mask, predicted_mask, average="binary", zero_division=0)
        precision, recall, _ = precision_recall_curve(true_mask, predicted_mask)
        ap = auc(recall, precision)
        
        scores.append([name, acc_value, f1_value, jac_value, recall_value, precision_value, ap])

    # Output mean scores
    mean_scores = np.mean([s[1:] for s in scores], axis=0)
    mean_prediction_time = np.mean(prediction_times)
    results_file = open(os.path.join(output_folder, "Evaluation_results.txt"), 'w')
    results_file.write(f"Mean Accuracy: {mean_scores[0]:0.5f}\n")
    results_file.write(f"Mean F1 Score: {mean_scores[1]:0.5f}\n")
    results_file.write(f"Mean Jaccard: {mean_scores[2]:0.5f}\n")
    results_file.write(f"Mean Recall: {mean_scores[3]:0.5f}\n")
    results_file.write(f"Mean Precision: {mean_scores[4]:0.5f}\n")
    results_file.write(f"Mean Average Precision (AP): {mean_scores[5]:0.5f}\n")
    results_file.write(f"Mean Prediction Time: {mean_prediction_time:.3f} seconds\n")
    results_file.close()


    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide the arguments for test data folder and true mask data")
        sys.exit(1)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_folder = sys.argv[1]
    true_mask_folder = sorted(glob(os.path.join(sys.argv[2], "*")))
    output_folder = 'EVA_YOLO'

    model = YOLO('customModel.pt')
    evaluate_model(model, test_data_folder, true_mask_folder, output_folder)
