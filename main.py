import config
from model import SSD_Model
from prepare import prepare_dataset
from train import retrain
from evaluate import evaluate

labels = [
    "person", "chair", "car", "dog", "bottle", "cat", "bird", "pottedplant", "sheep", "boat",
    "aeroplane", "tvmonitor", "sofa", "bicycle", "horse", "diningtable", "motorbike", "cow", "train", "bus"
]
label_amount = len(labels)

input_shape = (300, 300, 3)


if __name__ == "__main__":
    model = SSD_Model(input_shape, label_amount)

    training_dataset, testing_dataset = prepare_dataset("dataset", labels, training_ratio=config.TRAINING_SPLIT)
    retrain(model, training_dataset, config.TRAINING_ITERATIONS, config.EPOCHS, saved_ratio=config.SAVING_RATIO)

    model.save_model("model")
    model.plot_metrics()

    model.convert_to_mlmodel(labels)

    # _, testing_dataset = prepare_dataset("dataset", labels, training_ratio=0)

    mAP = evaluate(model, testing_dataset, labels)
    print(f"The model got an mAP score of {mAP}")

    metadata_changes = {
        "additional": {
            "Iterations trained": str(len(model.metrics["loss"])),
            "mAP": str(round(mAP, 5))
        }
    }
    model.save_mlmodel(metadata_changes=metadata_changes)
