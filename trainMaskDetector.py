# import the necessary packages
from keras.applications import MobileNetV2
from model.fcheadnet import FCHeadNet
from keras.applications import imagenet_utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from config import training_config as config
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the list of image paths and random shuffle the dataset
print("[INFO] loading images...")
imagePaths = list(paths.list_images(config.DATASET))

# Initialize the list of data and labels
dataset = []
labels = []
classNames = ["with_mask", "without_mask"]

preprocess = imagenet_utils.preprocess_input

# loop over the images
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    # load and preprocess the input image while ensuring
    # the VGG19 network accepts 224x224 images
    image = load_img(imagePath, target_size=config.INPUT_SHAPE)
    image = img_to_array(image)
    image = preprocess(image)

    dataset.append(image)
    labels.append(label)

# convert the dataset and labels to numpy arrays
dataset = np.array(dataset, dtype="float32")
labels= np.array(labels)

# convert the labels to vectors 
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training testing splits using
# 75% of the data for training and the remaning 25% for testing
(trainX, testX, trainY, testY) = train_test_split(dataset, labels, test_size=0.25, stratify=labels, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, 
width_shift_range=0.1, height_shift_range=0.1, 
shear_range=0.2, zoom_range=0.2, 
horizontal_flip=True, fill_mode="nearest")

# initialize the optimizer and the models for fine-tuning
print("[INFO] building model...")
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
headModel = FCHeadNet.build(baseModel, config.NUM_CLASSES, 128)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all the layers in base model and freeze them
# so they will not be updated during training
for layer in baseModel.layers:
    layer.trainable = False

# compile the model
print("[INFO] compiling model...")
opt = Adam(lr=config.INIT_LR, decay=config.INIT_LR/config.TRAIN_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics="accuracy")

# train the head of the network for a few epochs
model.fit_generator(aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE), validation_data=(testX, testY), epochs=config.TRAIN_EPOCHS, 
steps_per_epoch=len(trainX)//config.BATCH_SIZE, verbose=1)

# evaluate the network after initialization
print("[INFO] evaluating after initialization...")
predictions = model.predict(testX, batch_size=config.BATCH_SIZE)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

# now that the head FC layer have been trained/initialized, let's unfreeze
# the final set of CONV layer and make them trainable
for layer in baseModel.layers:
    layer.trainable=True

# for the changes to the model, recompilation is needed
print("[INFO] re-compiling model...")
opt = SGD(lr=config.INIT_LR, decay=config.INIT_LR/config.FINETUNE_EPOCHS, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics="accuracy")

# train the model again, this time fine-tuning both the final set of CONV layers
# along with the new FC layers
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE), validation_data=(testX, testY), epochs=config.FINETUNE_EPOCHS,
steps_per_epoch=len(trainX)//config.BATCH_SIZE, verbose=1)

# evaluate the network on the fine-tuned model
print("[INFO] evaluating after fine-tuning...")
predictions = model.predict(testX, batch_size=config.BATCH_SIZE)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

# save the model to disk
print("[INFO] serializing network...")
model.save(config.OUTPUT_MODEL, save_format="h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,config.FINETUNE_EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,config.FINETUNE_EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,config.FINETUNE_EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,config.FINETUNE_EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Flowers-17")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()