from ultralytics import YOLO

import config as cfg

if __name__ == "__main__":
    model = YOLO(cfg.model)

    model.train(
        resume=cfg.resume_training,
        data=cfg.data_path,
        cfg=cfg.config_path,
        epochs=cfg.training_epochs,
        imgsz=cfg.image_dim,
        batch=cfg.batch_spec,
        save_period=cfg.save_period,
        device=-1,
        workers=cfg.workers,
        seed=42,
        deterministic=False,
        rect=True,  # incompatible with DataLoader shuffle (it groups images by aspect ratio)
        project=cfg.save_path,
        name=cfg.model_name,
        plots=True,
        save=True
    )
    trainer = model.trainer
    trainer.save_checkpoint(trainer.epoch + 1, final=True)  # save the last checkpoint (ultralytics saves at the beginning of the next epoch it seems)

    """
    results = model("test.png")
    results[0].save(filename=f"result.png")
    """

    model.export(
        format=cfg.export_format,
        imgsz=cfg.image_size,
        half=cfg.export_half,
        nms=cfg.export_nms,
        conf=cfg.default_conf,
        iou=cfg.default_iou
    )
