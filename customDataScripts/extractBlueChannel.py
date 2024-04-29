from PIL import Image
import os
import argparse

# Function to extract and save the blue channel while preserving the folder structure
def extract_and_save_blue_channel(input_folder, output_folder):
    # Iterate through images in the input folder and its subfolders
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Open the image
                image_path = os.path.join(root, filename)
                img = Image.open(image_path)

                # Split the image into channels (R, G, B)
                r, g, b = img.split()

                # Determine the subfolder structure within the output folder
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)

                # Create the subfolder if it doesn't exist
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                # Save the blue channel in the corresponding subfolder with the original filename
                b.save(os.path.join(output_subfolder, filename))

                print(f"Extracted blue channel from {filename} and saved it to {output_subfolder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and save the blue channel from images')
    parser.add_argument('input_folder', help='Input folder containing images')
    parser.add_argument('output_folder', help='Output folder for extracted blue channel images')

    args = parser.parse_args()

    extract_and_save_blue_channel(args.input_folder, args.output_folder)