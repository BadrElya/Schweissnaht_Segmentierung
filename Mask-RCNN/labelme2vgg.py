import json
import os
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[1]
output_filename = sys.argv[2]

annotations = {}

for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        with open(os.path.join(input_dir, filename), 'r') as f:
            json_data = json.load(f)

        image_name = filename.split('.')[0]
        image_filename = os.path.join("", image_name + '.png')
        if 'imageHeight' in json_data:
            image_height = json_data['imageHeight']
        else:
            image_height = 0
        if 'imageWidth' in json_data:
            image_width = json_data['imageWidth']
        else:
            image_width = 0
        if 'shapes' in json_data:
            shapes = json_data['shapes']
        else:
            shapes = []

        image_annotations = []

        for shape in shapes:
            label = shape['label']
            points = shape['points']

            vgg_points = []
            for x, y in points:
                vgg_points.append(x)
                vgg_points.append(y)

            image_annotations.append({
                'shape_attributes': {
                    'name': 'polyline',
                    'all_points_x': vgg_points[::2],
                    'all_points_y': vgg_points[1::2]
                },
                'region_attributes': {
                    'names': label
                }
            })

        annotations[image_filename] = {
            'filename': image_filename,
            'size': image_width * image_height,
            'regions': image_annotations,
            'file_attributes': {}
        }

with open(os.path.join(output_dir, output_filename), 'w') as f:
    json.dump(annotations, f)
