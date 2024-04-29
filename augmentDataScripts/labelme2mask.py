import argparse
import glob
import os
import os.path as osp
import sys

import numpy as np

import labelme

def main(input_dir, output_root_dir):
    class_name_to_id = {}

    os.makedirs(output_root_dir, exist_ok=True)  # Create the root output directory if it doesn't exist

    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                json_file_path = osp.join(root, filename)
                print("Generating dataset from:", json_file_path)

                label_file = labelme.LabelFile(filename=json_file_path)

                base = osp.splitext(osp.basename(json_file_path))[0]

                # Create the subfolder structure in "Masks" to mirror "Images"
                output_subdir = osp.relpath(root, input_dir)
                output_dir = osp.join(output_root_dir, output_subdir)
                os.makedirs(output_dir, exist_ok=True)

                out_png_file = osp.join(output_dir, base + ".png")

                img = labelme.utils.img_data_to_arr(label_file.imageData)

                lbl, _ = labelme.utils.shapes_to_label(
                    img_shape=img.shape,
                    shapes=label_file.shapes,
                    label_name_to_value={'__ignore__': -1, '_background_': 0, 'weld': 1},
                )
                labelme.utils.lblsave(out_png_file, lbl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert labelme JSON files to masks')
    parser.add_argument('input_dir', help='Input directory containing labelme JSON files')
    parser.add_argument('output_root_dir', help='Root output directory for masks')

    args = parser.parse_args()

    main(args.input_dir, args.output_root_dir)
