import coremltools as ct
import datetime

import config as cfg
from files import load_yaml

class MLModelModifier:
    def __init__(self, model_path):
        self.mlmodel = ct.models.MLModel(model_path)

    def change_nms_pickTop(self):
        spec = self.mlmodel._spec  # mlmodel.get_spec()
        spec.pipeline.models[-1].nonMaximumSuppression.pickTop.perClass = False
        self.mlmodel._spec = spec

    def change_metadata(self, epochs):
        data_yaml = load_yaml(cfg.data_path)
        hyp_yaml = load_yaml(cfg.config_path)
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = {
            "short_description": f"Ultralytics YOLOv11l model trained on {data_yaml['path']}.",
            "author": "Mikael de Verdier",
            "additional": {
                "epoch": str(epochs),
                "date": date,
                "hyp": str(hyp_yaml)
            }
        }

        self.mlmodel.short_description = metadata["short_description"]
        self.mlmodel.author = metadata["author"]
        self.mlmodel.license = ""  # can i keep ultralytics'?
        self.mlmodel.version = ""

        self.mlmodel.user_defined_metadata.clear()
        self.mlmodel.user_defined_metadata.update(metadata.get("additional", {}))

    def save(self, output_path):
        self.mlmodel.save(output_path)
