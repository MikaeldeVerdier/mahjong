import config
from model import SSD_Model
from prepare import prepare_dataset
from train import retrain
from evaluate import evaluate

labels = [
    "Bamboo 1", "Bamboo 2", "Bamboo 3", "Bamboo 4", "Bamboo 5", "Bamboo 6", "Bamboo 7", "Bamboo 8", "Bamboo 9",
    "Dot 1", "Dot 2", "Dot 3", "Dot 4", "Dot 5", "Dot 6", "Dot 7", "Dot 8", "Dot 9",
    "Character 1", "Character 2", "Character 3", "Character 4", "Character 5", "Character 6", "Character 7", "Character 8", "Character 9",
    "East Wind", "South Wind", "West Wind", "North Wind",
    "Red Dragon", "Green Dragon", "White Dragon",
    "East Flower", "South Flower", "West Flower", "North Flower",
    "East Season", "South Season", "West Season", "North Season",
    "Back"
]
label_amount = len(labels)

input_shape = (512, 288, 3)


if __name__ == "__main__":
    model = SSD_Model(input_shape, label_amount)

    training_dataset, testing_dataset = prepare_dataset("ssd/dataset/data/train", labels, training_ratio=config.TRAINING_SPLIT)
    retrain(model, training_dataset, config.TRAINING_ITERATIONS, config.EPOCHS, saved_ratio=config.SAVING_RATIO)

    model.save_model("model")
    model.plot_metrics()

    model.convert_to_mlmodel(labels)

    # _, testing_dataset = prepare_dataset("ssd/dataset/data/test", labels, training_ratio=0)

    mAP = evaluate(model, testing_dataset, labels, AP_type="integration")
    print(f"The model got an mAP score of {mAP}")

    metadata_changes = {
        "additional": {
            "Iterations trained": str(len(model.metrics["loss"])),
            "mAP": str(round(mAP, 5))
        }
    }
    model.save_mlmodel(metadata_changes=metadata_changes)
