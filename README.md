# Hand-Gesture-Recognition

The different functions used in the code and their descriptions are as follows:

def getClassLabel(c_):
This function is used to return the class of the detected hand posture. The function waits for a fixed posture to appear for a certain number of frames(threshold) before returning
the class.

def camera_sim_realtime(model, dim_=(128, 96)):
This function is used to capture real time images through the webcam for classification into one of the classes.

def create_model(root="../models/"):
This function creates the Convolutional Model we used for our task. We create a CNN with three convolutional layers followed by a ReLU activation layer for each. The second convolutional layer is followed by a round of max-pooling and dropout with a dropout rate of 0.25. We flatten the output of the third convolution layer and connect it to a dense layer. This is followed by another ReLu activation layer , another dropout with a dropout rate of 0.5 . Finally this is connected to an output layer with n nodes (where n is the number of classes).

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],border_mode='valid',input_shape=input_shape))
→ adds the 1st convolutional layermodel.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
→ adds the second convolution ayer
model.add(Activation('relu'))
→ adds a ReLu activation function

model.add(MaxPooling2D(pool_size=pool_size))
→ adds a max pooling layer to reduce dimensionality model.add(Dropout(0.25))
→ adds dropout with dropout rate of 0.25

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
→ The Adam optimizer is used to optimize the model.

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
→ fits the model The function also saves the model so it can retrieve it later instead of retraining if a model file exists.

def setup():
This function takes the input and reshapes it. It also performs cross validation and training and testing of the dataset. One version of the code does this while another version
uses the webcam for testing.

Experimental Platform
Language:​ Python 2.7
Library:​ Keras, OpenCv, numpy, scipy
Machine:
Cpu​ : Intel i5 Processor
Ram​ : 8 GB
Graphics Card:​ Nvidia 980 GTX [4GB]
Run time:
In Machine with graphics Card:
Training​ : 40 Minutes [For 100 epochs]
Testing​ : in Seconds.In Machine without graphics Card:
Training: 6 Hours [For 100 epochs]
Testing: in Seconds.
