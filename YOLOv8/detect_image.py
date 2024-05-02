import argparse
from ultralytics import YOLO

def predict_with_args(image_path, model_path):
    model = YOLO(model_path)
    model.predict(source=image_path, show=True, save=True, hide_labels=False, hide_conf=False, conf=0.5, save_txt=False, save_crop=False, line_thickness=2)

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO prediction")

    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLO model")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    predict_with_args(args.image_path, args.model_path)
