# U-Net using TensorFlow 2 - Semantic Segmentation

GitHub Repository: [U-Net TensorFlow Repository](https://github.com/CatchZeng/tensorflow-unet-labelme)

Blog Tutorial: [Understanding the Code - U-Net Semantic Segmentation](https://makeoptim.com/en/deep-learning/yiai-unet/)

**Note**: The training module was originally a Jupyter Notebook file and has been converted into a Python script.

## Environment Setup

Before using U-Net for semantic segmentation, you need to set up the environment. Follow these steps:

1. Navigate to the U-Net directory:
   ```
   cd Path_To\Schweissnaht_Segmentierung\U-Net
   ```

2. Create a conda environment with Python 3.11.3 and install the required packages:
   ```
   conda create -n unet -y python=3.11.3 && conda activate unet && pip install --force-reinstall -r requirements.txt
   ```

3. Install `imgviz`:
   ```
   pip install imgviz
   ```

4. Install `labelme`:
   ```
   pip install labelme
   ```
## Dataset Structure

### LabelMe Annotated Data:

```
Dataset/
├── train
│   ├── img_0.png
│   ├── img_0.json
│   ├── img_1.png
│   ├── img_1.json
│   ├── img_2.png
│   ├── img_2.json
│   ├── img_100.png
│   ├── img_100.json
│   ├── img_101.png
│   ├── img_101.json
│   ├── img_102.png
│   ├── img_102.json
│   ├── ...
├── labels.txt
```

## Annotation Conversion

Prepare your dataset and convert annotations by following these steps:

1. Create a `labels.txt` file with the labels in dataset folder, similar to the example here: [LabelMe Labels](https://github.com/wkentaro/labelme/tree/main/examples/semantic_segmentation).

2. Convert LabelMe format to VOC format for your training dataset:
   ```
   python labelme2voc.py Dataset\train Dataset\train_voc --labels Dataset\labels.txt
   ```

3. The script `voc_annotation.py` automates the creation of text files that list filenames, effectively organizing a dataset into training and validation data..
   ```
   python voc_annotation.py
   ```

### Generated Data in VOC Format:

Ensure your Dataset is structured as follows:

```
Dataset/
├── train
├── labels.txt
├── train_voc
│   ├── ImageSets
│   │   ├── Segmentation
│   │   │   ├── test.txt
│   │   │   ├── train.txt
│   │   │   ├── trainval.txt
│   │   │   ├── val.txt
│   ├── JPEGImages
│   │   ├── img_0.jpg
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   ├── ...
│   ├── SegmentationClass
│   │   ├── img_0.npy
│   │   ├── img_1.npy
│   │   ├── img_2.npy
│   │   ├── ...
│   ├── SegmentationClassPNG
│   │   ├── img_0.png
│   │   ├── img_1.png
│   │   ├── img_2.png
│   │   ├── ...
│   ├── SegmentationClassVisualization
│   │   ├── img_0.jpg
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   ├── ...
│   ├── class_names.txt
```

## Training the Model

Train your U-Net model with the following command:
```
python train.py
```

## Image Segmentation

Perform image segmentation using your trained model with the following command:
```
python detect_image.py --image_path "../Testdata/Testdaten_G/1706039423.png" --model_path "customModel.h5"
```

## Evaluating the Model

This is for evaluating the U-Net model. It takes a folder containing the test data and their corresponding ground truth masks as input:

```
python unet_evaluation.py "../Testdata/Testdaten_G" "../Testdata/Testdaten_Masks"
```

These instructions will help you set up and use U-Net for semantic segmentation on your custom dataset. For more details, please refer to the provided blog tutorial and the GitHub repository.