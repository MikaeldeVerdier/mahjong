import random
import numpy as np

import config
from prepare import prepare_training

def retrain(model, dataset, iteration_amount, epochs, saved_ratio=1):
    for i in range(1, iteration_amount + 1):
        batches = random.sample(dataset, config.BATCH_SIZE)
        prepared_batches = [prepare_training(batch_img_path, batch_boxes, batch_labels, model.input_shape, model.class_amount, model.default_boxes) for batch_img_path, batch_boxes, batch_labels in batches]
        x, y = zip(*prepared_batches)

        model.train(np.array(x), np.array(y), epochs)

        if not int(i  % (iteration_amount / 10)):
            print(f"Training iteration {i} completed!")

        if not int(i % (iteration_amount * saved_ratio)):
            model.save_model("model")
            model.plot_metrics()

            print(f"Model saved at iteration {i}!")
