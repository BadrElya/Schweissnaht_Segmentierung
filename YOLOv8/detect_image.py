from ultralytics import YOLO

model = YOLO("customModel.pt")

model.predict(source="../Test_Images/1.png", show= True, save=True, hide_labels=False, hide_conf=False, conf=0.5, save_txt=False, save_crop=False, line_thickness=2) 