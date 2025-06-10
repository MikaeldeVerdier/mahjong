# Model configuration
model = "yolo11l.pt"
image_size = (512, 288)

# Training configuration
resume_training = False
data_path = "yolo/dataset/data.yaml"
config_path = "yolo/dataset/hyp.yaml"
image_dim = image_size[0]  # (1024, 576)  # Becomes rectangular anyway by rect=True (uses aspect ratio of images)
training_epochs = 2
batch_spec = 2  # Batch size || Fraction of cuda memory to use
save_period = 0
workers = 4
model_name = "yolov11-mahjong"
save_path = "yolo_save"

# Export configuration
export_format = "mlmodel"
export_half = True
export_nms = True
default_conf = 0.25
default_iou = 0.45
