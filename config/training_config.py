from os import path

DATASET = "../../datasets/Facial-mask"

OUTPUT_MODEL = "maskDetector.model"

NUM_CLASSES = 2

INPUT_SHAPE = (224, 224)

INIT_LR = 0.001
BATCH_SIZE = 32
TRAIN_EPOCHS = 20
FINETUNE_EPOCHS = 15