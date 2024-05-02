# YOLOv8 using TensorFlow 2 - Instance Segmentation

Licence: [Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)

Watch the video tutorial to understand the code and setup: [YOLOv8 Instance Segmentation Tutorial](https://www.youtube.com/watch?v=DMRlOWfRBKU)

For more detailed documentation, refer to the Ultralytics website: [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/tasks/segment/#dataset-format)

## Environment Setup

Before using YOLOv8 for instance segmentation, make sure to create and set up the environment. Follow these steps:

1. Navigate to the YOLOv8 directory:
   ```
   cd Path_To\Schweissnaht_Segmentierung\YOLOv8
   ```

2. Create a conda environment with Python 3.10 and install the required packages from `requirements.txt`: 
   ```
   conda create -n yolov8 -y python=3.10 && conda activate yolov8 && pip install -r requirements.txt
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
│   ├── ...
```
## Annotation Conversion

You can convert LabelMe annotations to YOLO format using the following script:

   Convert annotations in the training dataset with LabelMe format to YOLO format:
   ```
   labelme2yolo --json_dir dataset/train
   ```
### YOLOv8 Data Format:

Ensure your dataset is organized as follows:

```
Dataset/
├── train
│   ├── images
│   │   ├── img_0.jpg
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   ├── ...
│   ├── labels
│   │   ├── img_0.txt
│   │   ├── img_1.txt
│   │   ├── img_2.txt
│   │   ├── ...
│   ├── labels.cache
├── val
│   ├── images
│   │   ├── img_100.jpg
│   │   ├── img_101.jpg
│   │   ├── img_102.jpg
│   │   ├── ...
│   ├── labels
│   │   ├── img_100.txt
│   │   ├── img_101.txt
│   │   ├── img_102.txt
│   │   ├── ...
│   ├── labels.cache
```

## Training the Model

To train your YOLOv8 model for instance segmentation, use the following command:

```
yolo task=segment mode=train epochs=100 data=dataset.yaml model=yolov8n-seg.pt imgsz=512 batch=8
```

## Image Segmentation

Perform instance segmentation using your trained model with the following command:

```
python detect_image.py --image_path "../Testdata/Testdaten_G/1706039423.png" --model_path "customModel.pt"
```

## Retraining the Model

If needed, you can retrain your model using the following command:

```
yolo task=detect mode=train resume model=runs\segment\train\weights\last.pt data=dataset.yaml epochs=100 imgsz=512 batch=8
```

## Evaluating the Model

This is for evaluating the YOLO model. It takes a folder containing the test data and their corresponding ground truth masks as input:

```
python yolo_evaluation.py "../Testdata/Testdaten_G" "../Testdata/Testdaten_Masks"
```

Follow these instructions to set up and use YOLOv8 for instance segmentation on your custom dataset. You can find additional details in the video tutorial and the provided documentation links.