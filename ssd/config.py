# Training configuration
TRAINING_SPLIT = 1
USE_AUGMENTATION = True
LEARNING_RATE = {0: 1e-3, 100_000: 1e-4, 150_000: 1e-5}
MOMENTUM = 0.9
ALPHA = 1
L2_REG = 5e-4
HARD_NEGATIVE_RATIO = 3
VARIANCES = [0.1, 0.1, 0.2, 0.2]
BATCH_SIZE = 16
TRAINING_ITERATIONS = 500_000 - 294_720
EPOCHS = 1
VALIDATION_SPLIT = 0.2

# Saving configuration
SAVE_FOLDER_PATH = "save_folder"
SAVING_RATIO = 0.05

# MODEL INFO:
# Based on Version 2

# DATASET INFO:
# Synthetic data: 100k images (newly generated)
