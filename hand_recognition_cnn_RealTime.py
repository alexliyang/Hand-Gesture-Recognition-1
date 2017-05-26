import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from load_dataset import load_data
from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD, Adam
from keras.models import model_from_json
from core import Globals
import os
import cv2
from Action import Action

np.random.seed(1337)  # To get tge testing data randomly

batch_size = 128
nb_classes = 5
nb_epoch = 12

# Directory location of Train and Test Datasets
loc_ = "../Custom/"

img_rows, img_cols = 96, 128
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
# Cross Validation
X_train, X_test, Y_train, Y_test = (None, None, None, None)
input_shape = None
frameCount = 0
setLabel = 0
threshold = 10

def getClassLabel(c_):

	global frameCount, setLabel

	if setLabel != c_:
		frameCount = 0
	setLabel = c_

	if c_ == 0:
		frameCount = (frameCount + 1)
		if frameCount > threshold:
			return "One Finger Only"
	if c_ == 1:
		frameCount = (frameCount + 1)
		if frameCount > threshold:
			return "Victory"
	if c_ == 2:
		frameCount = (frameCount + 1)
		if frameCount > threshold:
			Action.play()
			return "Fist"
	if c_ == 3:
		frameCount = (frameCount + 1)
		if frameCount > threshold:
			Action.pause()
			return "Wait"
	if c_ == 4:
		if frameCount > 20:
			return "Nothing There"

#Simulating the Camera on real time
def camera_sim_realtime(model, dim_=(128, 96)):

	cap = cv2.VideoCapture(0)
	while True:
		r_, frame = cap.read()
		frame = cv2.flip(frame, 1)
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		keyPress = cv2.waitKey(1) & 0xFF

		if keyPress == ord('q'):
			break
		elif keyPress == ord('c') or True:
			print "Web Cam Image Capturing!"
			captureImg = cv2.resize(frame_gray, dim_)
			xx = captureImg.astype('float32')
			xx /= 255
			xx = np.reshape(xx, (1, 1, dim_[1], dim_[0]))
			classes = model.predict(xx)
			x, y = (100, 100)
			cv2.putText(frame, getClassLabel(np.argmax(classes)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
		cv2.imshow(Globals.windowTitle, frame)


def create_model(root="../models/"):
	filename = 'hand_recognition_model.json'

	#If the model file already exists in ../Model, then do not train it again, only load the model
	if os.path.exists(root + filename):
		json_file = open(root + filename, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights(root + 'hand_recognition_model_weights.h5')

		adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
		loaded_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

		return loaded_model

	#Add components sequentially
	model = Sequential()
	
	#Add a 2D convolution layer with defined kernel Size
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
							border_mode='valid',
	                        input_shape=input_shape))
	#Set Relu as activation function
	model.add(Activation('relu'))

	#Add another 2D convolution layer with defined kernel Size
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
	model.add(Activation('relu'))

	#Add a max pool layer to reduce its dimetionality
	#For each of the regions represented by the filter, we will take 
	#the max of that region and create a new, output matrix where each 
	#element is the max of a region in the original input.
	model.add(MaxPooling2D(pool_size=pool_size))

	#Perform drop out.
	#Make 25% value zero.
	model.add(Dropout(0.25))

	#Added another convolution Layer
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
	model.add(Activation('relu'))
	
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

  
	
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
	
	#Compile the model with cross entropy loss
	model.compile(loss='categorical_crossentropy',
	              optimizer=adam,
	              metrics=['accuracy'])
	
	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
	
	model.save(root + 'hand_recognition_model.h5')
	model.save_weights(root + 'hand_recognition_model_weights.h5')

	return model

def setup():

	global X_train, X_test, Y_train, Y_test, input_shape
	X, y = load_data(loc_)
	X_train = X
	y_train = y

	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	if K.image_dim_ordering() == 'th':
		X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
		# X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
		# X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)

	X_train = X_train.astype('float32')
	X_train = X_train / 255

	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)

setup()
model = create_model()
print X_train.shape
#starting the real time camera simulation
camera_sim_realtime(model)
