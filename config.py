# Training configuration
TRAINING_SPLIT = 1
AUGMENTATION_AMOUNT = 5
LEARNING_RATE = 1e-5
MOMENTUM = 0.9
L2_REG = 5e-4
HARD_NEGATIVE_RATIO = 3
SQ_VARIANCES = [0.1, 0.1, 0.2, 0.2]
BATCH_SIZE = 32
TRAINING_ITERATIONS = 8
EPOCHS = 1
VALIDATION_SPLIT = 0.2

# Saving configuration
SAVE_FOLDER_PATH = "save_folder"
SAVING_FREQUENCY = int(TRAINING_ITERATIONS / 8)
