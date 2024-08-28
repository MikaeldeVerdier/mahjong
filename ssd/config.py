# Training configuration
TRAINING_SPLIT = 1
USE_AUGMENTATION = True
LEARNING_RATE = {0: 1e-3, 80_000: 1e-4, 130_000: 1e-5}
MOMENTUM = 0.9
ALPHA = 1
L2_REG = 5e-4
HARD_NEGATIVE_RATIO = 3
VARIANCES = [0.1, 0.1, 0.2, 0.2]
BATCH_SIZE = 16
TRAINING_ITERATIONS = 150_000
EPOCHS = 1
VALIDATION_SPLIT = 0.2

# Saving configuration
SAVE_FOLDER_PATH = "save_folder"
SAVING_RATIO = 0.05

# DATASET INFO:
# Synthetic data: 12k images (same dataset as Version 2, only clipped at 12k (in order of annotations file))
# Real data: 12.291k images (only 11.976k used due to ignore_no_gt in preparation being True)
