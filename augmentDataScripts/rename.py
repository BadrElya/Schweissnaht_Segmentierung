import os
import json
import shutil
import argparse

def rename_images_and_update_json(image_folder, output_folder):
    # List all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]  # Adjust the extension as needed

    # Sort the image files to maintain order
    image_files.sort()

    # Create a new folder to store renamed images and JSON files
    os.makedirs(output_folder, exist_ok=True)

    # Initialize a counter for renaming
    counter = 1

    # Iterate through the image files and process JSON files
    for image_file in image_files:
        # Generate the new image filename with the counter
        new_image_filename = f"{counter}.png"
        new_image_path = os.path.join(output_folder, new_image_filename)

        # Process the corresponding JSON LabelMe file
        json_filename = image_file.replace(".png", ".json")
        old_json_path = os.path.join(image_folder, json_filename)
        new_json_path = os.path.join(output_folder, f"{counter}.json")

        # Load the JSON data
        with open(old_json_path, "r") as json_file:
            data = json.load(json_file)

        # Update the filename in JSON data
        data["imagePath"] = new_image_filename

        # Save the updated JSON data
        with open(new_json_path, "w") as new_json_file:
            json.dump(data, new_json_file, indent=4)

        # Move the image file to the output folder (overwriting if it already exists)
        shutil.move(os.path.join(image_folder, image_file), new_image_path)

        # Increment the counter
        counter += 1

    print("Image renaming and JSON file updates completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename images and update JSON files')
    parser.add_argument('image_folder', help='Directory containing image files')
    parser.add_argument('output_folder', help='Directory for renamed images and updated JSON files')

    args = parser.parse_args()

    rename_images_and_update_json(args.image_folder, args.output_folder)