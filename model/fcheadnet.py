from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.convolutional import AveragePooling2D

class FCHeadNet:
    @staticmethod
    def build(baseModel, classes, D): # D is the number of nodes in the fully connected layer
        # initialize the head model that will be placed on top of the base, then add a FC layer
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(7,7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        
        # add a softmax layer
        headModel = Dense(classes, activation="softmax")(headModel)

        return headModel 