import os
import sys
import numpy as np
import skimage.io
import skimage.color
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, precision_recall_curve, auc

from mrcnn import model as modellib
from mrcnn.config import Config

class CustomConfig(Config):
    NAME = "object"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  # Background + object
    DETECTION_MIN_CONFIDENCE = 0.3

class InferenceConfig(CustomConfig):
    DETECTION_NMS_THRESHOLD = 0.3

def visualize_masks(image_path, ground_truth_mask, predicted_mask, output_path):
    image = Image.open(image_path).convert("RGBA")

    # Farben für Ground Truth und Vorhersagen
    gt_color = (255, 0, 0, 125)  # Rot mit Transparenz
    pred_color = (0, 255, 0, 125)  # Grün mit Transparenz

    # Konvertieren der NumPy-Arrays in PIL-Images
    gt_mask_image = Image.fromarray((ground_truth_mask * 255).astype(np.uint8))
    pred_mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))

    # Erstellen von transparenten Ebenen für Ground Truth- und vorhergesagte Masken
    gt_mask = Image.new('RGBA', image.size, (0, 0, 0, 0))
    pred_mask = Image.new('RGBA', image.size, (0, 0, 0, 0))

    # Zeichnen der Masken
    gt_mask.paste(gt_color, (0, 0), gt_mask_image)
    pred_mask.paste(pred_color, (0, 0), pred_mask_image)

    # Kombinieren der Originalbild- und Maskenbilder
    blended_image = Image.alpha_composite(image, gt_mask)
    blended_image = Image.alpha_composite(blended_image, pred_mask)

    # Speichern des kombinierten Bildes
    blended_image.save(output_path)

def evaluate_model(model, test_data_folder, true_mask_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Dummy prediction for warm-up
    dummy_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    model.detect([dummy_image])

    scores = []
    prediction_times = []

    with open(os.path.join(output_folder, "Evaluation_results.txt"), 'w') as results_file:
        true_mask_paths = sorted([os.path.join(true_mask_folder, f) for f in os.listdir(true_mask_folder) if f.endswith('.png')])
        for true_path in tqdm(true_mask_paths):
            name = os.path.basename(true_path)
            image_path = os.path.join(test_data_folder, name)
            image = skimage.io.imread(image_path)
            if image.ndim == 2:
                image = skimage.color.gray2rgb(image)
            elif image.ndim == 3 and image.shape[2] == 4:
                image = image[..., :3]

            start_time = time.time()
            results = model.detect([image])
            end_time = time.time()

            prediction_time = end_time - start_time
            prediction_times.append(prediction_time)

            r = results[0]

            true_mask = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)
            true_mask = (true_mask / 255.0 > 0.5).astype(np.uint8)

            if r['masks'].size > 0:
                highest_score_index = np.argmax(r['scores'])
                pred_mask = r['masks'][:, :, highest_score_index]
                pred_mask = (pred_mask > 0).astype(np.uint8)
            else:
                # Use a zero mask if no prediction is available
                pred_mask = np.zeros_like(true_mask)

            # Calculate metrics
            acc_value = accuracy_score(true_mask.flatten(), pred_mask.flatten())
            f1_value = f1_score(true_mask.flatten(), pred_mask.flatten(), average="binary")
            jac_value = jaccard_score(true_mask.flatten(), pred_mask.flatten(), average="binary")
            recall_value = recall_score(true_mask.flatten(), pred_mask.flatten(), average="binary")
            precision_value = precision_score(true_mask.flatten(), pred_mask.flatten(), average="binary", zero_division=0)
            precision, recall, _ = precision_recall_curve(true_mask.flatten(), pred_mask.flatten())
            ap = auc(recall, precision)

            # Save results to list
            scores.append([name, acc_value, f1_value, jac_value, recall_value, precision_value, ap, prediction_time])

            # Visualize and save image with masks
            output_image_path = os.path.join(output_folder, name)
            visualize_masks(image_path, true_mask, pred_mask, output_image_path)

        # Calculate mean scores and prediction time
        mean_scores = np.mean([s[1:-1] for s in scores], axis=0)
        mean_prediction_time = np.mean(prediction_times)

        # Write mean scores and prediction time to file
        results_file.write(f"Mean Accuracy: {mean_scores[0]:0.5f}\n")
        results_file.write(f"Mean F1 Score: {mean_scores[1]:0.5f}\n")
        results_file.write(f"Mean Jaccard: {mean_scores[2]:0.5f}\n")
        results_file.write(f"Mean Recall: {mean_scores[3]:0.5f}\n")
        results_file.write(f"Mean Precision: {mean_scores[4]:0.5f}\n")
        results_file.write(f"Mean Average Precision (AP): {mean_scores[5]:0.5f}\n")
        results_file.write(f"Mean Prediction Time: {mean_prediction_time:.3f} seconds\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide the arguments Test Data Folder and Test Data Masks Folder.")
        sys.exit(1)
    test_data_folder = sys.argv[1]
    true_mask_folder = sys.argv[2]
    deck = sys.argv[2]
    output_folder = 'EVA_MaskRCNN/'
    modelName = 'customModel.h5'
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir="logs")
    model.load_weights(modelName, by_name=True)

    
    evaluate_model(model, test_data_folder, true_mask_folder, output_folder)
