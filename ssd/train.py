import random
import numpy as np

import config
from prepare import prepare_training

def get_lr(iteration):
    highest_iter = -1
    highest_val = 1e-10  # 0 ?

    for (key, val) in config.LEARNING_RATE.items():  # listcomp instead?
        if highest_iter < key < iteration:
            highest_val = val

    return highest_val


def retrain(model, dataset, start_iteration, end_iteration, epochs, saved_ratio=1):
    used_start_iteration = 0
    if start_iteration is None and len(list(model.metrics.values())) != 0:
        used_start_iteration = len(list(model.metrics.values())[0])

    model.model.optimizer.lr = get_lr(used_start_iteration)

    iteration_amount = end_iteration - used_start_iteration

    for i in range(used_start_iteration, end_iteration):
        if i in config.LEARNING_RATE.keys():
            model.model.optimizer.lr = config.LEARNING_RATE[i]

        batches = random.sample(dataset, config.BATCH_SIZE)
        prepared_batches = [prepare_training(batch_img_path, batch_boxes, batch_labels, model.input_shape, model.class_amount, model.default_boxes) for batch_img_path, batch_boxes, batch_labels in batches]
        x, y = zip(*prepared_batches)

        model.train(np.array(x), np.array(y), epochs)

        if not int(i % (iteration_amount / 10)):
            print(f"Training iteration {i} completed!")

        if i != 0 and not int(i % (iteration_amount * saved_ratio)):
            model.save_model("model")
            model.plot_metrics()

            print(f"Model saved at iteration {i}!")
