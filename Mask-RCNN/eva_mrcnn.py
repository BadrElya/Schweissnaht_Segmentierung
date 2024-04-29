import os
import numpy as np
import skimage.io
import cv2
from mrcnn import model as modellib
from mrcnn.config import Config
from mrcnn import visualize
import time
import json
from PIL import Image, ImageDraw

class CustomConfig(Config):
    NAME = "object"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  # Background + Weld
    DETECTION_MIN_CONFIDENCE = 0.9

class InferenceConfig(CustomConfig):
    DETECTION_NMS_THRESHOLD = 0.3
    
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0

def precision_recall(true_mask, pred_mask):
    TP = np.sum(np.logical_and(pred_mask, true_mask))
    FP = np.sum(np.logical_and(pred_mask, ~true_mask))
    FN = np.sum(np.logical_and(~pred_mask, true_mask))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return precision, recall

def global_ap_voc(all_precisions, all_recalls):
    recall_levels = np.linspace(0, 1, 11)
    precisions_at_recall = []
    sorted_indices = np.argsort(all_recalls)
    sorted_recalls = all_recalls[sorted_indices]
    sorted_precisions = all_precisions[sorted_indices]
    for recall_level in recall_levels:
        precisions = sorted_precisions[sorted_recalls >= recall_level]
        max_precision = max(precisions) if len(precisions) > 0 else 0
        precisions_at_recall.append(max_precision)
    ap = np.mean(precisions_at_recall)
    return ap

def load_labelme_masks(annotation_path, image_shape):
    with open(annotation_path, 'r') as file:
        data = json.load(file)
    mask = Image.new('1', (image_shape[1], image_shape[0]), 0)
    for shape in data['shapes']:
        polygon = shape['points']
        polygon = [tuple(map(int, point)) for point in polygon]
        ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
    return np.array(mask, dtype=bool)

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

def evaluate_model(model, test_data_folder, output_folder):
    ious = []
    precisions = []
    recalls = []
    prediction_times = []

    os.makedirs(output_folder, exist_ok=True)

    # Load the model and perform dummy predictions to warm-up
    dummy_image = np.zeros((256, 256, 3), dtype=np.uint8)
    _ = model.detect([dummy_image])

    with open(os.path.join(output_folder, 'evaluation_results.txt'), 'w') as file, open(os.path.join(output_folder, 'metrics_results.txt'), 'w') as metrics_file:
        file.write("Filename, IoU, Precision, Recall, Prediction Time\n")

        for filename in os.listdir(test_data_folder):
            if filename.endswith(".png"):
                image_path = os.path.join(test_data_folder, filename)
                annotation_path = os.path.join(test_data_folder, filename.replace(".png", ".json"))

                # Load image and predict
                image = skimage.io.imread(image_path)
                if image.ndim == 2:
                    image = skimage.color.gray2rgb(image)
                elif image.ndim == 3 and image.shape[2] == 4:
                    image = image[..., :3]

                # Record prediction time
                start_time = time.time()
                
                results = model.detect([image])
                r = results[0]

                # Calculate prediction time
                end_time = time.time()
                prediction_time = end_time - start_time
                prediction_times.append(prediction_time)
                
                # Create Ground-Truth mask according to the annotation data using load_labelme_masks function
                true_mask = load_labelme_masks(annotation_path, image.shape)

                # Crop predicted mask to the same size as the true mask
                pred_mask = r['masks'][:, :true_mask.shape[0], :true_mask.shape[1]]
                pred_mask = pred_mask[:, :, 0]
                
                # Calculate metrics
                iou = calculate_iou(true_mask, pred_mask)
                precision, recall = precision_recall(true_mask, pred_mask)
                ious.append(iou)
                precisions.append(precision)
                recalls.append(recall)

                # Write results to output file
                file.write(f"{filename}, {iou:.4f}, {precision:.4f}, {recall:.4f}, {prediction_time:.2f}\n")

                # Save combined image with masks
                save_path = os.path.join(output_folder, filename.replace('.png', '_visualized.png'))
                visualize_masks(image_path, true_mask, pred_mask, save_path)
                print(f"Combined image saved to {save_path}")

        # Calculate AP and average prediction time
        ap = global_ap_voc(np.array(precisions), np.array(recalls))
        mean_iou = np.mean(ious)
        mean_prediction_time = np.mean(prediction_times)
        
        # Write mean IoU, AP, and Mean Prediction Time to metrics file
        metrics_file.write(f"Average Precision (AP): {ap:.4f}\n")
        metrics_file.write(f"Mean IoU (mIoU): {mean_iou:.4f}\n")
        metrics_file.write(f"Mean Prediction Time: {mean_prediction_time:.2f} seconds\n")


if __name__ == "__main__":
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir="logs")
    model.load_weights('maskrcnnmodel.h5', by_name=True)
    test_data_folder = 'Testdaten'
    output_folder = 'EVA_MaskRCNN'
    evaluate_model(model, test_data_folder, output_folder)
