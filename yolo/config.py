# Model configuration
model = "yolo_save/yolov11-mahjong_NEW slow/weights/epoch195.pt"
image_size = (512, 288)

# Training configuration
resume_training = False
data_path = "yolo/dataset/data.yaml"
config_path = "yolo/dataset/hyp.yaml"
image_dim = image_size[0]  # (512, 288)  # Becomes rectangular anyway by rect=True (uses aspect ratio of images)
training_epochs = 5000  # won't use all epochs but better to be too high than too low
batch_spec = 38  # Batch size || Fraction of cuda memory to use
save_period = 5
workers = 8
model_name = "Yolo2"
save_path = "yolo_save"

# Export configuration
export_format = "mlmodel"
export_half = True
export_nms = True
default_conf = 0.25
default_iou = 0.45
