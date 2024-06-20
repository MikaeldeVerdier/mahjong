import config
from model import SSD_Model
from prepare import prepare_dataset
from train import retrain
from evaluate import evaluate

labels = ["aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"]
label_amount = len(labels)

input_shape = (300, 300, 3)


if __name__ == "__main__":
    model = SSD_Model(input_shape, label_amount)

    training_dataset, testing_dataset = prepare_dataset("dataset/data/train", labels, training_ratio=config.TRAINING_SPLIT)
    retrain(model, training_dataset, config.TRAINING_ITERATIONS, config.EPOCHS, saved_ratio=config.SAVING_RATIO)

    model.save_model("model")
    model.plot_metrics()

    model.convert_to_mlmodel(labels)

    # _, testing_dataset = prepare_dataset("dataset/data/test", labels, training_ratio=0)

    mAP = evaluate(model, testing_dataset, labels)
    print(f"The model got an mAP score of {mAP}")

    metadata_changes = {
        "additional": {
            "Iterations trained": str(len(model.metrics["loss"])),
            "mAP": str(round(mAP, 5))
        }
    }
    model.save_mlmodel(metadata_changes=metadata_changes)
