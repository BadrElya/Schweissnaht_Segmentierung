import argparse
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def cvtColor(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def detect_image(image_path, model_path):
    image = Image.open(image_path)
    image = cvtColor(image)
    old_img = image.copy()
    ori_h, ori_w = image.size[1], image.size[0]

    image_data, nw, nh = resize_image(image, (512, 512))
    image_data = normalize(np.array(image_data, dtype=np.float32))
    image_data = np.expand_dims(image_data, 0)

    model = tf.keras.models.load_model(model_path)

    pr = model.predict(image_data)[0]
    pr = pr[(512 - nh) // 2:(512 - nh) // 2 + nh, (512 - nw) // 2:(512 - nw) // 2 + nw]
    pr = cv2.resize(pr, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
    pr = pr.argmax(axis=-1)

    # Find contours
    contours, _ = cv2.findContours(pr.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.array(old_img)
    cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)  # Draw red contours

    final_image = Image.fromarray(contour_img)

    # Save the image
    save_path = 'annotated_image.png'
    final_image.save(save_path)
    print(f"Image saved to {save_path}")

    return final_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Detection')
    parser.add_argument('--image_path', type=str, help='Path to the input image')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    args = parser.parse_args()

    if args.image_path and args.model_path:
        detect_image(args.image_path, args.model_path)
    else:
        print("Please provide both --image_path and --model_path arguments.")
