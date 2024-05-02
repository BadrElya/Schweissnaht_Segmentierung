import os
import json
import argparse

# Function to process a single LabelMe JSON file
def process_labelme_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    if 'imagePath' in data:
        image_path = data['imagePath']
        if image_path.lower().endswith('.png'):
            new_image_path = image_path[:-4] + '.jpg'
            data['imagePath'] = new_image_path

            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
                print(f"Modified: {file_path}")

# Function to iterate through files in a directory
def process_labelme_files_in_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                process_labelme_file(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process LabelMe JSON files in a directory')
    parser.add_argument('directory_path', help='Directory path containing LabelMe JSON files')

    args = parser.parse_args()

    process_labelme_files_in_directory(args.directory_path)
