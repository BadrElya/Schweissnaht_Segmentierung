# Image Augmentation Pipeline

The augmentation of the images is done in three steps. First, masks are generated from the original JSON files of the images. These masks and the original images are augmented in the same way based on their names. Finally, the generated augmented masks are used to create the corresponding JSON files. These generated JSON files are then named to match the augmented original images.


## 1. labelme2mask.py

This script processes JSON files from a specified Images directory and generates corresponding Masks in a subdirectory structure.

```
python labelme2mask.py Images Masks
```

## 2. augment.py

This Python script performs image augmentation on two sets of images and saves the augmented images. The script includes various augmentation techniques like horizontal and vertical flips, rotations, and distortion. After running the script, you will have augmented images in the specified output folders, organized by the type of augmentation applied.

```
python augment.py Images/1 Masks/1 augmented_Images/1 augmented_Masks/1
```

## 3. mask2labelme.py

This Python script processes images and creates corresponding LabelMe JSON annotation files. After running this script, you will have a set of JSON annotation files corresponding to the mask images in the specified directory.

```
python mask2labelme.py "augmented_Masks\\1" "*.png"
```

## 4. cpyAnnotationMasks2Images.py

This script is designed to copy JSON files from a specified input folder to a designated output folder while preserving the directory structure.

```
python cpyAnnotationMasks2Images.py "augmented_Masks" "augmented_Images"
```