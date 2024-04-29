# Scripts Descriptions

## 1. updateImageDataAndWidth.py

This script processes JSON files in a specified directory and its subdirectories, and it specifically removes the image data and sets the image width to 1248 in these JSON files.

```
python updateImageDataAndWidth.py Images
```

## 2. resize.py

This script resizes images to a specified dimension and updates the corresponding annotation data to match the new image size.

```
python resize.py -t ./train
```

- `-t` specifies the path to the images and labeling files.

## 3. padding_images.py

This script takes a directory of image files, resizes them by adding black padding to the right side if needed, and saves the resized images back to their original file paths.

```
python padding_images.py Images
```

## 4. extractBlueChannel.py

This script processes images in the input_folder, extracts the blue channel from each image, preserves the folder structure, and saves the blue channel images in the output_folder.

```
python extractBlueChannel.py Images Images_BC
```

## 5. correctjpg2png.py

This script updates the "imagePath" format in LabelMe JSON annotation files from PNG to JPG.

```
python correctjpg2png.py Images
```

## 6. convert2png.py

This script converts images to JPG format and saves the converted images in the specified output directory.

```
python convert2png.py Images converted_PNG_Images
```