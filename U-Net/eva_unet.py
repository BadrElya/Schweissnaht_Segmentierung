import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import cv2
import os
import time

def normalize(image):
    return image / 127.5 - 1

def resize_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image, nw, nh

def create_dummy_image(size):
    return Image.new('RGB', size, color='white')

def load_labelme_masks(annotation_path, image_shape):
    with open(annotation_path, 'r') as file:
        data = json.load(file)
    mask = Image.new('1', (image_shape[1], image_shape[0]), 0)
    for shape in data['shapes']:
        polygon = shape['points']
        polygon = [tuple(map(int, point)) for point in polygon]
        ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
    return np.array(mask, dtype=bool)

def precision_recall(true_mask, pred_mask):
    TP = np.sum(np.logical_and(pred_mask == 1, true_mask == 1))
    FP = np.sum(np.logical_and(pred_mask == 1, true_mask == 0))
    FN = np.sum(np.logical_and(pred_mask == 0, true_mask == 1))
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

    # Erstellen der Legende
    legend = Image.new('RGBA', (200, 100), (255, 255, 255, 0))
    draw = ImageDraw.Draw(legend)
    draw.rectangle([10, 10, 30, 30], fill=gt_color)
    draw.text((40, 15), "Ground Truth", fill='white')
    draw.rectangle([10, 40, 30, 60], fill=pred_color)
    draw.text((40, 45), "Prediction", fill='white')

    # Hinzufügen der Legende zum kombinierten Bild
    blended_image.paste(legend, (40, 40), legend)

    # Speichern des kombinierten Bildes
    blended_image.save(output_path)


def evaluate_model(model, test_data_folder, output_folder):
    ious = []
    all_precisions = []
    all_recalls = []
    prediction_times = []

    os.makedirs(output_folder, exist_ok=True)

    # Dummy-Bild erstellen
    dummy_image = create_dummy_image((512, 512))
    dummy_image = normalize(np.array(dummy_image, dtype=np.float32))
    dummy_image = np.expand_dims(dummy_image, 0)
    _ = model.predict(dummy_image)

    with open(os.path.join(output_folder, 'evaluation_results.txt'), 'w') as file, open(os.path.join(output_folder, 'metrics_results.txt'), 'w') as metrics_file:
        file.write("Filename, IoU, Precision, Recall, Prediction Time\n")

        for filename in os.listdir(test_data_folder):
            if filename.endswith(".png"):
                image_path = os.path.join(test_data_folder, filename)
                annotation_path = os.path.join(test_data_folder, filename.replace(".png", ".json"))

                # Image processing
                image = Image.open(image_path)
                ori_h, ori_w = image.size[1], image.size[0]
                image_data, nw, nh = resize_image(image, (512, 512))
                image_data = normalize(np.array(image_data, dtype=np.float32))
                image_data = np.expand_dims(image_data, 0)

                # Model prediction
                start_time = time.time()  # Startzeit
                pr = model.predict(image_data)[0]
                end_time = time.time()  # Endzeit
                duration = end_time - start_time  # Dauer der Vorhersage
                print(f"Prediction time for {filename}: {duration:.2f} seconds")
                prediction_times.append(duration)
                pr = pr[(512 - nh) // 2:(512 - nh) // 2 + nh, (512 - nw) // 2:(512 - nw) // 2 + nw]
                pr = cv2.resize(pr, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
                pr = pr.argmax(axis=-1)

                # Ground truth and predicted mask
                ground_truth_mask = load_labelme_masks(annotation_path, (ori_h, ori_w))
                predicted_mask = pr > 0

                # Metrics calculation
                intersection = np.logical_and(ground_truth_mask, predicted_mask)
                union = np.logical_or(ground_truth_mask, predicted_mask)
                iou = np.sum(intersection) / np.sum(union)
                ious.append(iou)
                precision, recall = precision_recall(ground_truth_mask, predicted_mask)
                all_precisions.append(precision)
                all_recalls.append(recall)

                # Visualize masks
                output_path = os.path.join(output_folder, filename.replace(".png", "_visualized.png"))
                visualize_masks(image_path, ground_truth_mask, predicted_mask, output_path)

                # Write evaluation results to file
                file.write(f"{filename}, {iou:.4f}, {precision:.4f}, {recall:.4f}, {duration:.2f}\n")

        mean_iou = np.mean(ious)
        mean_prediction_time = np.mean(prediction_times)
        ap = global_ap_voc(np.array(all_precisions), np.array(all_recalls))

        # Write mean IoU, Global AP and Mean Prediction Time to file
        metrics_file.write(f"Average Precision (AP): {ap:.4f}\n")
        metrics_file.write(f"Mean IoU (mIoU): {mean_iou:.4f}\n")
        metrics_file.write(f"Mean Prediction Time: {mean_prediction_time:.2f}\n")

    print(f"Mean IoU: {mean_iou:.4f}, Global AP: {ap:.4f}, Mean Prediction Time: {mean_prediction_time:.2f}")

if __name__ == "__main__":
    model_path = 'unetmodel.h5'
    model = tf.keras.models.load_model(model_path)
    test_data_folder = 'Testdaten'
    output_folder = 'EVA_UNET'
    evaluate_model(model, test_data_folder, output_folder)
