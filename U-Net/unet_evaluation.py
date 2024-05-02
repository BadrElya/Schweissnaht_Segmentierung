import os
import sys
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, precision_recall_curve, auc

def normalize(image):
    """Normalize the image to the range [-1, 1]."""
    return image / 127.5 - 1

def create_dummy_image(size):
    return Image.new('RGB', size, color='white')

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
    
def predict_and_process(model, image_path, model_size=(512, 512)):
    image = Image.open(image_path)
    ori_h, ori_w = image.size[1], image.size[0]
    image_data, nw, nh = resize_image(image, model_size)
    image_data = normalize(np.array(image_data, dtype=np.float32))
    image_data = np.expand_dims(image_data, 0)
    start_time = time.time()
    pr = model.predict(image_data)[0]
    end_time = time.time()
    prediction_time = end_time - start_time
    # Adjust the prediction to only consider the visible part of the resized image
    pr = pr[(model_size[1] - nh) // 2:(model_size[1] - nh) // 2 + nh, (model_size[0] - nw) // 2:(model_size[0] - nw) // 2 + nw]
    pr = cv2.resize(pr, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)

    # Assuming class 1 is the class of interest
    pr = pr.argmax(axis=-1)
    binary_mask = (pr == 1).astype(np.uint8)  # Mask for class 1, without inverting colors

    # Check if the binary mask is completely empty
    if np.sum(binary_mask) == 0:
        # Create an empty mask of the same size
        binary_mask = np.zeros_like(binary_mask)

    return binary_mask.flatten(), ori_w, ori_h, prediction_time

def evaluate_model(model, test_data_folder, true_mask_paths, output_folder):
    # Dummy-Bild erstellen
    dummy_image = create_dummy_image((512, 512))
    dummy_image = normalize(np.array(dummy_image, dtype=np.float32))
    dummy_image = np.expand_dims(dummy_image, 0)
    _ = model.predict(dummy_image)
    
    os.makedirs(output_folder, exist_ok=True)
    scores = []

    for true_path in tqdm(true_mask_paths):
        name = os.path.basename(true_path)
        image_path = os.path.join(test_data_folder, name)
        pred_y, orig_w, orig_h, prediction_time = predict_and_process(model, image_path)

        true_y = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)
        true_y = (true_y / 255.0 > 0.5).astype(np.int32).flatten()

        acc_value = accuracy_score(pred_y, true_y)
        f1_value = f1_score(pred_y, true_y, average="binary")
        jac_value = jaccard_score(pred_y, true_y, average="binary")
        recall_value = recall_score(pred_y, true_y, average="binary")
        precision_value = precision_score(pred_y, true_y, average="binary", zero_division=0)
        precision, recall, _ = precision_recall_curve(true_y, pred_y)
        ap = auc(recall, precision)

        scores.append([name, acc_value, f1_value, jac_value, recall_value, precision_value, ap, prediction_time])

        # Visualize and save the overlayed masks
        visualize_masks(image_path, true_y.reshape(orig_h, orig_w), pred_y.reshape(orig_h, orig_w), os.path.join(output_folder, name))

    mean_scores = np.mean([s[1:] for s in scores], axis=0)
    mean_prediction_time = np.mean([s[7] for s in scores])  # Berechnen des mittleren Vorhersagezeit

    results_file = open(os.path.join(output_folder, "Evaluation_results.txt"), 'w')
    results_file.write(f"Mean Accuracy: {mean_scores[0]:0.5f}\n")
    results_file.write(f"Mean F1 Score: {mean_scores[1]:0.5f}\n")
    results_file.write(f"Mean Jaccard: {mean_scores[2]:0.5f}\n")
    results_file.write(f"Mean Recall: {mean_scores[3]:0.5f}\n")
    results_file.write(f"Mean Precision: {mean_scores[4]:0.5f}\n")
    results_file.write(f"Mean Average Precision (AP): {mean_scores[5]:0.5f}\n")
    results_file.write(f"Mean Prediction Time: {mean_prediction_time:.3f} seconds\n")  # Richtig schreiben der mittleren Vorhersagezeit
    results_file.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python unet_evaluation.py <test_data_folder> <test_data_masks_folder>")
        sys.exit(1)

    test_data_folder = sys.argv[1]
    true_mask_paths = sorted(glob(os.path.join(sys.argv[2], "*")))
    output_folder = 'EVA_UNET'
    model_path = 'customModel.h5'
    model = tf.keras.models.load_model(model_path)
    evaluate_model(model, test_data_folder, true_mask_paths, output_folder)
