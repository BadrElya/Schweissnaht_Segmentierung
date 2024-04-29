import cv2
import sys, json
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
        if w != dim[0] or h != dim[1]:
            print(annotation_path, w, h, ' Resizing')
            nresize += 1
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            for indx, p in enumerate(jsndata['shapes']):
                points = [[i[0] / w, i[1] / h] for i in p['points']]
                points = [[i[0] * dim[0], i[1] * dim[1]] for i in points]
                jsndata['shapes'][indx]['points'] = points
                points = np.int32([points])
        else:
            print(annotation_path, w, h, ' Skip')
        cv2.imwrite(output_labled_path, img)
        jsndata['imageWidth'] = dim[0]
        jsndata['imageHeight'] = dim[1]
        if jsndata['imagePath'] != imgname:
            print('Error image name = ' + imgname + ' while json has ' + jsndata['imagePath'])
            print(annotation_path)
            exit()
        json.dump(jsndata, open(output_labled_path.replace('.png', '.json'), 'w'))

    print('Total number of resized images = ', nresize)

def main(input_folder, output_folder, dim):
    for root, dirs, files in os.walk(input_folder):
        for dir_name in dirs:
            input_subfolder = os.path.join(root, dir_name)
            output_subfolder = os.path.join(output_folder, dir_name)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)  # Create the output folder if it doesn't exist
            process_image(input_subfolder, output_subfolder, dim)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--path_imgs_annotation', required=True, help='path to imgs with annotations')
    args = ap.parse_args()

    dim = (512, 512)
    main(args.path_imgs_annotation, 'output', dim)