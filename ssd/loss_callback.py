from keras.callbacks import Callback

class LossHistory(Callback):
    def __init__(self, *losses):
        super().__init__()

        self.losses = losses

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for loss_var in self.losses:
            name = loss_var.name[:-2]  # a bit unreliable. could do "".join(loss_var.split(":")[:-1])
            logs[name] = float(loss_var.numpy())  # why don't these sum up to the total_loss (or val_loss), but close?
