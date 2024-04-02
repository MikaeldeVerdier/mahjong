import random
import numpy as np

import config

def retrain(model, dataset, iteration_amount, epochs, saved_ratio=1):
    for i in range(1, iteration_amount + 1):
        x, y = zip(*random.sample(dataset, config.BATCH_SIZE))

        model.train(np.array(x), np.array(y), epochs)

        if not int(i  % (iteration_amount / 10)):
            print(f"Training iteration {i} completed!")

        if not int(i % (iteration_amount * saved_ratio)):
            model.save_model("model")
            model.plot_metrics()

            print(f"Model saved at iteration {i}!")
