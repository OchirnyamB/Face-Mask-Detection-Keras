from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import imagenet_utils
from keras.models import load_model
from config import training_config as config
from preprocessing.aspectawarepreprocessor import AspectAwarePreProcessor
import numpy as np 
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# load the haar cascades
face_cascade = cv2.CascadeClassifier("faceDetector.xml")

# load an input image from disk
print("[INFO] loading the image...")
img = cv2.imread(args["image"])

# detect faces
faces = face_cascade.detectMultiScale(img, 1.1, 4)

# define preprocess function used to extend dimension for 
# the number of samples in the input shape
preprocess = imagenet_utils.preprocess_input
# initialize the image preprocessor and the list of RGB channel averages
aap = AspectAwarePreProcessor(224, 224)

# load the model
print("[INFO] loading the trained model ...")
model  = load_model("maskDetector.model")

for (x,y,w,h) in faces:
    # extract the face ROI, convert it from BGR to RGB channel
    # resize it to 224x224 and preprocess it
    face = img[x:x+w, y:y+h]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = aap.preprocess(face)
    face = img_to_array(face)
    face = preprocess(face)
    face = np.expand_dims(face, axis=0)

    # pass the face through the model to determine if the face 
    # has a mark or not
    print("[INFO] running prediction on the image ...")
    (mask, withoutMask) = model.predict(face)[0]
    label = "Mask" if mask > withoutMask else "No Mask"
    # put a box around the face, green if wearing a mask red if not
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    # include the probability in the label
    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

    cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(img, (x,y), ((x+w),(y+h)), color, 2)

# show the output image
cv2.imshow("Mask Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()