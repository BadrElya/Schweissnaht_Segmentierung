import glob
from natsort import natsorted
import os
import cv2
import numpy as np
import json
import argparse

def main(dataset_dir, extension):
    images = natsorted(glob.glob(os.path.join(dataset_dir, extension)))

    epsilon = 1

    for i in range(len(images)):
        name = images[i].strip(".png") + ".json"
        labelme_data = {
            "version": "4.5.10",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(images[i]),
            "imageData": None,
            "imageHeight": 1248,
            "imageWidth": 1248,
        }

        mask = cv2.imread(images[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) < 3:
                continue
            polygon = cv2.approxPolyDP(contour, epsilon, True).squeeze().tolist()
            labelme_shape = {
                "label": "weld",
                "points": polygon,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
            labelme_data["shapes"].append(labelme_shape)

        # Save LabelMe JSON file
        with open(name, "w") as json_file:
            json.dump(labelme_data, json_file, indent=2)

        if i % 100 == 0:
            print(i / len(images))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert image masks to LabelMe JSON files')
    parser.add_argument('dataset_dir', help='Directory containing image masks')
    parser.add_argument('extension', help='File extension for image masks (e.g., "*.png")')

    args = parser.parse_args()

    main(args.dataset_dir, args.extension)