# Training configuration
TRAINING_SPLIT = 1
USE_AUGMENTATION = True
LEARNING_RATE = {0: 1e-3, 500_000: 1e-3, 800_000: 1e-5}
MOMENTUM = 0.9
ALPHA = 1
L2_REG = 5e-4
HARD_NEGATIVE_RATIO = 3
VARIANCES = [0.1, 0.1, 0.2, 0.2]
BATCH_SIZE = 8
START_ITERATION = None  # None means it will use the length of the models first metric
END_ITERATION = 1_000_000
EPOCHS = 1
VALIDATION_SPLIT = 0.2

# Saving configuration
SAVE_FOLDER_PATH = "save_folder"
SAVING_RATIO = 0.05

# MODEL INFO:
# Fresh start

# DATASET INFO:
# Synthetic data: 50k images (newly generated)
