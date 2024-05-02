import cv2
import sys
import json
import argparse
from xml.dom import minidom
import numpy as np
import os

def process_image(input_folder, output_folder, dim):
    images_names = [img for img in os.listdir(input_folder) if img.endswith(".png")]

    nresize = 0
    for imgname in images_names:
        imgpath = os.path.join(input_folder, imgname)
        annotation_path = os.path.join(input_folder, imgname.replace('.png', '.json'))
        if not os.path.isfile(annotation_path):
            continue
        jsndata = json.load(open(annotation_path, 'r'))
        output_labled_path = os.path.join(output_folder, imgname)
        img = cv2.imread(imgpath)
        h, w = img.shape[0], img.shape[1]

        # img = cv2.resize(img, (dim[1], dim[0]), interpolation=cv2.INTER_AREA)
        if w != dim[1] or h != dim[0]:
            print(annotation_path, w, h, ' Resizing')
            nresize += 1
            img = cv2.resize(img, (dim[1], dim[0]), interpolation=cv2.INTER_AREA)
            for indx, p in enumerate(jsndata['shapes']):
                points = [[i[0] / w, i[1] / h] for i in p['points']]
                points = [[i[0] * dim[1], i[1] * dim[0]] for i in points]
                jsndata['shapes'][indx]['points'] = points
                points = np.int32([points])
        else:
            print(annotation_path, w, h, ' Skip')
        cv2.imwrite(output_labled_path, img)
        jsndata['imageWidth'] = dim[1]
        jsndata['imageHeight'] = dim[0]
        if jsndata['imagePath'] != imgname:
            print('Error image name = ' + imgname + ' while json has ' + jsndata['imagePath'])
            print(annotation_path)
            exit()
        json.dump(jsndata, open(output_labled_path.replace('.png', '.json'), 'w'))

    print('Total number of resized images = ', nresize)

def main(input_folder, output_folder, dim):
    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist.")
        return

    # Überprüfen, ob der Eingabeordner Unterverzeichnisse hat
    subdirectories = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]

    # Wenn der Eingabeordner keine Unterverzeichnisse hat, durchsuchen Sie den Elternordner
    if not subdirectories:
        print(f"No subdirectories found in {input_folder}.")
        for root, dirs, files in os.walk(os.path.dirname(input_folder)):
            for dir_name in dirs:
                input_subfolder = os.path.join(root, dir_name)
                output_subfolder = os.path.join(output_folder, os.path.relpath(input_subfolder, os.path.dirname(input_folder)))
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)  # Create the output folder if it doesn't exist
                process_image(input_subfolder, output_subfolder, dim)
    else:
        # Durchsuchen Sie die Unterverzeichnisse des Eingabeordners
        for root, dirs, files in os.walk(input_folder):
            for dir_name in dirs:
                input_subfolder = os.path.join(root, dir_name)
                output_subfolder = os.path.join(output_folder, os.path.relpath(input_subfolder, input_folder))
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)  # Create the output folder if it doesn't exist
                process_image(input_subfolder, output_subfolder, dim)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--path_imgs_annotation', required=True, help='path to imgs with annotations')
    ap.add_argument('-w', '--width', type=int, default=512, help='Width of the image')
    ap.add_argument('-H', '--height', type=int, default=512, help='Height of the image')
    args = ap.parse_args()

    dim = (args.height, args.width)
    main(args.path_imgs_annotation, 'resized_input_folder', dim)
