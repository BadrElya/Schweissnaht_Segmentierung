import os
from PIL import Image
import argparse

def convert_images_to_png(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # List all image files in the input directory and its subdirectories
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    for root, _, files in os.walk(input_directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))

    # Convert and save each image to PNG format
    for image_file in image_files:
        try:
            image = Image.open(image_file)
            output_file = os.path.join(output_directory, os.path.basename(image_file))
            output_file = os.path.splitext(output_file)[0] + ".png"  # Change file extension to .png
            image.save(output_file, 'png')
            print(f"Converted: {image_file} -> {output_file}")
        except Exception as e:
            print(f"Error converting {image_file}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert images to PNG format')
    parser.add_argument('input_directory', help='Input directory containing images')
    parser.add_argument('output_directory', help='Output directory for converted PNGs')

    args = parser.parse_args()

    convert_images_to_png(args.input_directory, args.output_directory)