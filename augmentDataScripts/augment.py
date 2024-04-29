import os
import cv2
import numpy as np
import random
import argparse

def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder, filename))
            images[filename] = img
    return images

def save_augmented_images(output_folder, images):
    os.makedirs(output_folder, exist_ok=True)
    for filename, img in images.items():
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)

def augment_images(images1, images2, output_folder1, output_folder2):
    augmentation_types = ['horizontal_flip', 'vertical_flip', 'rotated_positive', 'rotated_negative', 'distorted']

    augmented_images1 = {}
    augmented_images2 = {}

    # Shuffle the list of filenames to maintain correspondence
    filenames1 = list(images1.keys())
    random.shuffle(filenames1)
    filenames2 = list(images2.keys())
    random.shuffle(filenames2)

    for filename1 in filenames1:
        print("Generating dataset from:", filename1)
        img1 = images1[filename1]
        img2 = images2[filename1]

        # Horizontal flip
        horizontal_flip1 = cv2.flip(img1, 1)
        horizontal_flip2 = cv2.flip(img2, 1)

        # Vertical flip
        vertical_flip1 = cv2.flip(img1, 0)
        vertical_flip2 = cv2.flip(img2, 0)

        # Rotation
        rotation_matrix_positive = cv2.getRotationMatrix2D(
            (img1.shape[1] / 2, img1.shape[0] / 2), 5, 1)
        rotation_matrix_negative = cv2.getRotationMatrix2D(
            (img1.shape[1] / 2, img1.shape[0] / 2), -5, 1)
        rotated_positive1 = cv2.warpAffine(img1, rotation_matrix_positive, (img1.shape[1], img1.shape[0]))
        rotated_negative1 = cv2.warpAffine(img1, rotation_matrix_negative, (img1.shape[1], img1.shape[0]))
        rotated_positive2 = cv2.warpAffine(img2, rotation_matrix_positive, (img2.shape[1], img2.shape[0]))
        rotated_negative2 = cv2.warpAffine(img2, rotation_matrix_negative, (img2.shape[1], img2.shape[0]))

        augmented_images1[f"{augmentation_types[0]}_{filename1}"] = horizontal_flip1
        augmented_images1[f"{augmentation_types[1]}_{filename1}"] = vertical_flip1
        augmented_images1[f"{augmentation_types[2]}_{filename1}"] = rotated_positive1
        augmented_images1[f"{augmentation_types[3]}_{filename1}"] = rotated_negative1

        augmented_images2[f"{augmentation_types[0]}_{filename1}"] = horizontal_flip2
        augmented_images2[f"{augmentation_types[1]}_{filename1}"] = vertical_flip2
        augmented_images2[f"{augmentation_types[2]}_{filename1}"] = rotated_positive2
        augmented_images2[f"{augmentation_types[3]}_{filename1}"] = rotated_negative2

        # Distortion (randomized)
        k = np.random.uniform(0.01, 0.05)  # Control the distortion strength
        n = np.random.randint(3, 5)  # Number of distortions
        for i in range(n):
            rows1, cols1, _ = img1.shape
            rows2, cols2, _ = img2.shape

            dx1 = np.random.randint(-10, 10)
            dy1 = np.random.randint(-10, 10)
            pts1 = np.float32([[dx1, dy1], [cols1 - dx1, dy1], [dx1, rows1 - dy1], [cols1 - dx1, rows1 - dy1]])
            pts2 = np.float32([[dx1, dy1], [cols2 - dx1, dy1], [dx1, rows2 - dy1], [cols2 - dx1, rows2 - dy1]])

            matrix1 = cv2.getPerspectiveTransform(pts1, pts2)
            matrix2 = cv2.getPerspectiveTransform(pts1, pts2)

            distorted1 = cv2.warpPerspective(img1, matrix1, (cols1, rows1))
            distorted2 = cv2.warpPerspective(img2, matrix2, (cols2, rows2))

            augmented_images1[f"{augmentation_types[4]}_{i}_{filename1}"] = distorted1
            augmented_images2[f"{augmentation_types[4]}_{i}_{filename1}"] = distorted2

    for augmentation_type in augmentation_types:
        save_augmented_images(os.path.join(output_folder1, augmentation_type), {
            k: v for k, v in augmented_images1.items() if k.startswith(augmentation_type)})
        save_augmented_images(os.path.join(output_folder2, augmentation_type), {
            k: v for k, v in augmented_images2.items() if k.startswith(augmentation_type)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment image data')
    parser.add_argument('folder1', help='Input directory for images 1')
    parser.add_argument('folder2', help='Input directory for images 2')
    parser.add_argument('output_folder1', help='Output directory for augmented images 1')
    parser.add_argument('output_folder2', help='Output directory for augmented images 2')

    args = parser.parse_args()

    images1 = load_images_from_folder(args.folder1)
    images2 = load_images_from_folder(args.folder2)

    augment_images(images1, images2, args.output_folder1, args.output_folder2)

    print("Augmentation completed for both folders with corresponding images.")