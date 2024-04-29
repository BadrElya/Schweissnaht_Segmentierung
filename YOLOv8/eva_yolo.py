import json
from ultralytics import YOLO
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageOps
import os
import numpy as np
import time

# Modell laden
model = YOLO("yolomodel.pt")

def draw_annotations(image_pil, annotation_path, color=(0, 255, 0, 125)):
    with open(annotation_path, 'r') as file:
        data = json.load(file)
    mask = Image.new('1', image_pil.size, 0)  # Erstelle eine neue Maske
    draw = ImageDraw.Draw(mask)
    draw_img = ImageDraw.Draw(image_pil, "RGBA")
    for shape in data['shapes']:
        polygon = shape['points']
        polygon = [tuple(map(int, point)) for point in polygon]
        draw_img.polygon(polygon, fill=color)  # Zeichne das Polygon in Gelb auf das Bild
        draw.polygon(polygon, outline=1, fill=1)  # Zeichne das Polygon auf die Maske für Metriken
    return np.array(mask, dtype=np.bool_)

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

output_folder = 'EVA_YOLO'
os.makedirs(output_folder, exist_ok=True)
test_data_folder = 'Testdaten'

ious = []
precisions = []
recalls = []
prediction_times = []

# Dummy-Bild erstellen
dummy_image = Image.new('RGB', (640, 480), color='white')

# Zeit für Vorhersage erfassen
start_time = time.time()

# Dummy-Bild vorhersagen
_ = model.predict(dummy_image)

# Zeit für Vorhersage berechnen
end_time = time.time()
dummy_prediction_time = end_time - start_time

# Öffnen der Ausgabedatei für das Schreiben
with open(os.path.join(output_folder, "evaluation_results.txt"), "w") as f, open(os.path.join(output_folder, "metrics_results.txt"), "w") as f_metrics:
    f.write("Filename, IoU, Precision, Recall, Prediction Time\n")
    for filename in os.listdir(test_data_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(test_data_folder, filename)
            annotation_path = os.path.join(test_data_folder, filename.replace(".png", ".json"))

            # Zeit für Vorhersage erfassen
            start_time = time.time()

            # Vorhersage durchführen
            results = list(model.predict(source=image_path, show=False, save=False, show_labels=False, show_conf=False, conf=0.5, save_txt=False, save_crop=False))
            result0 = results[0]

            # Zeit für Vorhersage berechnen
            end_time = time.time()
            prediction_time = end_time - start_time
            prediction_times.append(prediction_time)

            # Originalbild in ein PIL-Image umwandeln
            image_pil = T.ToPILImage()(result0.orig_img)

            # Maske aus YOLO-Ergebnis in Rot zeichnen
            pred_mask = Image.new('1', image_pil.size, 0)
            draw_pred = ImageDraw.Draw(pred_mask)
            for segment in result0.masks.xy:
                segment_list = [(int(x), int(y)) for x, y in segment]
                draw = ImageDraw.Draw(image_pil, "RGBA")
                draw.polygon(segment_list, fill=(255, 0, 0, 125))  # Rote Füllung mit Transparenz
                draw_pred.polygon(segment_list, outline=1, fill=1)  # Zeichne das Polygon auf die Maske für Metriken

            # LabelMe JSON-Annotationen in Gelb zeichnen und wahre Maske erstellen
            true_mask = draw_annotations(image_pil, annotation_path)

            # Berechnung der Metriken
            pred_mask = np.array(pred_mask, dtype=np.bool_)
            iou = calculate_iou(true_mask, pred_mask)
            precision, recall = precision_recall(true_mask, pred_mask)
            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)

            # Schreiben der Ergebnisse in die Datei
            f.write(f"{filename}, {iou:.4f}, {precision:.4f}, {recall:.4f}, {prediction_time:.2f}\n")

            # Bild speichern
            save_path = os.path.join(output_folder, filename)
            image_pil.save(save_path)
            print(f"Image saved to {save_path}")

    # Berechnung von AP und durchschnittlicher Vorhersagezeit
    ap = global_ap_voc(np.array(precisions), np.array(recalls))
    mean_prediction_time = np.mean(prediction_times)
    
    # Write mean IoU, AP, and Mean Prediction Time to metrics file
    f_metrics.write(f"Average Precision (AP): {ap:.4f}\n")
    f_metrics.write(f"Mean IoU (mIoU): {np.mean(ious):.4f}\n")
    f_metrics.write(f"Mean Prediction Time: {mean_prediction_time:.2f} seconds\n")
