import os
import json

# Function to normalize coordinates
def normalize_coordinates(x, y, width, height):
    return x / width, y / height

# Input and output directories
input_directory = "./Deck0008/train/labels"  # Replace with the path to your JSON files directory
output_directory = "./Deck0008/train/labels"  # Replace with the path where you want to save the txt files

# Iterate through JSON files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".json"):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, os.path.splitext(filename)[0] + ".txt")

        # Load JSON data
        with open(input_path, 'r') as json_file:
            data = json.load(json_file)

        # Extract image size
        image_width = data["imageWidth"]
        image_height = data["imageHeight"]

        # Extract and normalize points
        normalized_points = []
        for shape in data["shapes"]:
            if "points" in shape:
                points = shape["points"]
                normalized = [normalize_coordinates(x, y, image_width, image_height) for x, y in points]
                normalized_points.extend(normalized)

        # Write the normalized points to the output file
        with open(output_path, 'w') as txt_file:
            txt_file.write("0 " + " ".join([f"{x:.17f}" for point in normalized_points for x in point]))

        print(f"Converted {input_path} to {output_path}")

print("Conversion completed.")
