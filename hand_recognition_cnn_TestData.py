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
import os

def camera_sim_realtime(model, dim_=(128, 96)):
	
	cap = cv2.VideoCapture(0)
	while True:
		r_, frame = cap.read()
		frame = cv2.flip(frame, 1)
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# show live feed
		cv2.imshow(Globals.windowTitle, frame)
		keyPress = cv2.waitKey(1) & 0xFF

		if keyPress == ord('c'):
			print "Capturing Images from Webcam"
			captureImg = cv2.resize(frame_gray, dim_)
			xx = np.asarray(xx)
			xx = np.reshape(xx, dim_)
			print model.predict(xx)
			break
		elif keyPress == ord('q'):
			break

def create_model(root="./"):
	
	filename = 'hand_recognition_model.json'
	if os.path.exists(filename):
		json_file = open(root + filename, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights(root + 'hand_recognition_model_weights.h5')

		# compile model before use
		adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
		loaded_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

		return loaded_model

	model = Sequential()
	
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
	                        border_mode='valid',
	                        input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(0.25))

	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
 	model.add(Activation('relu'))
	
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
	
	model.compile(loss='categorical_crossentropy',
	              optimizer=adam,
	              metrics=['accuracy'])
	
	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
	
	model_json = model.to_json()
	with open(root + filename, "w") as json_file:
		json_file.write(model_json)
	
	model.save_weights(root + "hand_recognition_model_weights.h5")
	
	return model

#to generate testing data set randomly
np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_classes = 5
nb_epoch = 20

# Directory location of Train and Test Datasets
loc_ = "/home/samiran/Desktop/suhitCS726/Dataset-CS726"

img_rows, img_cols = 96, 128
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2,2)
# convolution kernel size
kernel_size = (3, 3)

X, y = load_data(loc_)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = create_model()
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
