from ultralytics import YOLO

import config as cfg
from model_plotter import ModelPlotter
from mlmodel_modifier import MLModelModifier

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

    """  # Plot results
    plotter = ModelPlotter(f"{cfg.save_path}/{cfg.model_name}/results.csv")
    plotter.create_results(model)
    plotter.plot_results()
    """

    """  # Test inference
    results = model("test.png")
    results[0].save(filename=f"result.png")
    """

    model_path = model.export(
        format=cfg.export_format,
        imgsz=cfg.image_size,
        half=cfg.export_half,
        nms=cfg.export_nms,
        conf=cfg.default_conf,
        iou=cfg.default_iou
    )

    model_modifier = MLModelModifier(model_path)
    model_modifier.change_nms_pickTop()
    model_modifier.change_metadata(model.trainer.epochs if model.trainer else model.ckpt.get("epoch", None))
    model_modifier.save(model_path)
