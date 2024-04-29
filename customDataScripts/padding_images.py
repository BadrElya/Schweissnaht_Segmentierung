import cv2
import os
import argparse

def list_images_in_directory(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_list = []

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_list.append(os.path.join(root, file))

    return image_list

def pad_image(image, target_size):
    h, w, c = image.shape
    if w < target_size:
        pad_width = target_size - w
        padded_image = cv2.copyMakeBorder(image, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        padded_image = image

    return padded_image

def main(root_directory):
    image_files = list_images_in_directory(root_directory)

    if not image_files:
        print("No images found in the specified directory and its subfolders.")
        return

    target_size = 1248

    for image_path in image_files:
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load image: {image_path}")
        else:
            # Pad the image at the end (right side)
            padded_image = pad_image(image, target_size)

            # Save the padded image back to the original file path
            cv2.imwrite(image_path, padded_image)
            print(f"Image saved with padding: {image_path}")

    print("Padding complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pad images in a directory')
    parser.add_argument('root_directory', help='Root directory containing images')

    args = parser.parse_args()

    main(args.root_directory)