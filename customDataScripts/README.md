# Scripts Descriptions

## 1. updateImageDataAndWidth.py

This script processes JSON files in a specified directory and its subdirectories, and it specifically removes the image data and sets the image width to 1248 in these JSON files.

```
python updateImageDataAndWidth.py input_folder 1248
```

## 2. padding_images.py

This script takes a directory of image files, resizes them by adding zero padding to the right side if needed, and saves the resized images back to their original file paths.

```
python padding_images.py input_folder --target_size 1248
```

## 3. resize.py

This script resizes images to a specified dimension and updates the corresponding annotation data to match the new image size.

```
python resize.py -t ./input_folder -w 512 -H 512
```

- `-t` specifies the path to the images and labeling files.

## 4. extractBlueChannel.py

This script processes images in the input_folder, extracts the blue channel from each image, preserves the folder structure, and saves the blue channel images in the output_folder.

```
python extractBlueChannel.py input_folder input_folder_BC
```

## 5. correctpng2jpg.py

This script updates the "imagePath" format in LabelMe JSON annotation files from PNG to JPG.

```
python correctpng2jpg.py input_folder
```

## 6. convert2png.py

This script converts images to PNG format and saves the converted images in the specified output directory.

```
python convert2png.py input_folder converted_PNG_input_folder
```