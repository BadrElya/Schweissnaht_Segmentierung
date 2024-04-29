# Mask R-CNN using TensorFlow 2 - Instance Segmentation

GitHub Repository: [Mask R-CNN TensorFlow Repository](https://github.com/AarohiSingla/Mask-R-CNN-using-Tensorflow2)

Video Tutorial: [Understanding the Code - Mask R-CNN Instance Segmentation](https://www.youtube.com/watch?v=QP9Nl-nw890)

## Environment Setup

Before using Mask R-CNN for instance segmentation, you need to set up the environment. Follow these steps:

1. Navigate to the Mask-RCNN directory:
   ```
   cd Path_To\Image_Segmentation\Mask-RCNN
   ```

2. Create a conda environment with Python 3.8.5 and install the required packages:
   ```
   conda create -n maskrcnn -y python=3.8.5 && conda activate maskrcnn && pip install --force-reinstall -r requirements.txt
   ```
   
## Create validation data

30% of the training data from the dataset/train folder will be moved to the dataset/validation folder

   ```
   python createValidationSet.py
   ```
   
## Annotation Conversion

If your image data is labeled with LabelMe, it needs to be converted using the "labelme2vgg.py" script:

1. Convert the training dataset with LabelMe format to VGG format:
   ```
   python labelme2vgg.py dataset/train train.json
   ```

2. Convert the validation dataset with LabelMe format to VGG format:
   ```
   python labelme2vgg.py dataset/validation validation.json
   ```

## Training the Model

Train your Mask R-CNN model with the following command:
```
python train.py
```

## Image Segmentation

Perform instance segmentation using your trained model with the following command:
```
python detect_image.py
```

## Dataset Structure

Ensure your dataset is structured as follows:

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
│   ├── ...
├── validation
│   ├── img_100.png
│   ├── img_100.json
│   ├── img_101.png
│   ├── img_101.json
│   ├── img_102.png
│   ├── img_102.json
│   ├── ...
├── labels.txt
```

### Generated Files in VGG Format:

```
Dataset/
├── train
│   ├── train.json
│   ├── img_0.png
│   ├── img_0.json
│   ├── img_1.png
│   ├── img_1.json
│   ├── img_2.png
│   ├── img_2.json
│   ├── ...
├── validation
│   ├── validation.json
│   ├── img_100.png
│   ├── img_100.json
│   ├── img_101.png
│   ├── img_101.json
│   ├── img_102.png
│   ├── img_102.json
│   ├── ...
```

These instructions will help you set up and use Mask R-CNN for instance segmentation on your custom dataset. For more details, please refer to the provided video tutorial and the GitHub repository.