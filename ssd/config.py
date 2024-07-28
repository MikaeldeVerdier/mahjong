# Training configuration
TRAINING_SPLIT = 0.7
USE_AUGMENTATION = True
LEARNING_RATE = {0: 1e-3, 5: 1e-4}
MOMENTUM = 0.9
ALPHA = 1
L2_REG = 5e-4
HARD_NEGATIVE_RATIO = 3
VARIANCES = [0.1, 0.1, 0.2, 0.2]
BATCH_SIZE = 32
TRAINING_ITERATIONS = 10
EPOCHS = 1
VALIDATION_SPLIT = 0.2

# Saving configuration
SAVE_FOLDER_PATH = "/Users/mikaeldeverdier/Downloads/save_folder 99.3 mj 35k"
SAVING_RATIO = 1
