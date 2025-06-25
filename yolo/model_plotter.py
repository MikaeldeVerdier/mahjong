from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.plotting import plot_results
from pathlib import Path
import yaml

class ModelPlotter:
    def plot_results(self):
        plot_results("yolo_save/yolov11-mahjong_NEW/results.csv")
        model = YOLO("yolo_save/yolov11-mahjong_NEW/weights/last.pt")

        opt_path = "yolo_save/yolov11-mahjong_NEW/args.yaml"

        # Load training args
        with open(opt_path, "r") as f:
            args = yaml.safe_load(f)

        args['resume'] = True  # makes sure it resumes from that folder
        args["device"] = "cpu"

        # Create a Trainer manually
        trainer = BaseTrainer(overrides=args)
        trainer.model = model

        # Now you can use Ultralytics' built-in plot method
        trainer.plot_metrics()
        pass

ModelPlotter().plot_results()