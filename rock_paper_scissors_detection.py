#Download zip files containing images from Laurence Moroney's Repository
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip \
    -O /content/Rock_Paper_Scissors_Training.zip
  
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip \
    -O /content/Rock_Paper_Scissors_Testing.zip

#Import for unzipping files
import zipfile

#Import for ImageDataGenerator and CNN modeling
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

#Import for plotting performance metrics
import matplotlib.pyplot as plt

#Unzip training and testing data
training_data = zipfile.ZipFile('Rock_Paper_Scissors_Testing.zip', 'r')
training_data.extractall('/content/')
training_data.close()
testing_data = zipfile.ZipFile('Rock_Paper_Scissors_Training.zip', 'r')
testing_data.extractall('/content/')
testing_data.close()

#Use ImageDataGenerator to rescale and augment images to increase volume and diversity of training data
training_datagen = ImageDataGenerator(
      rescale = 1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

#Use ImageDataGenerator to stream data and automatically label them based on folder names
training_generator = training_datagen.flow_from_directory(
	'/content/rps/',
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

#Use ImageDataGenerator to rescale validation data
validation_datagen = ImageDataGenerator(rescale = 1./255)

#Use ImageDataGenerator to stream data and automatically label them based on folder names
validation_generator = validation_datagen.flow_from_directory(
	'/content/rps-test-set/',
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

#Use TensorFlow and Keras to develop CNN
model = tf.keras.models.Sequential([

#This layer defines the input image and performs the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
#This layer pools the pixels by finding the max pixel in a 2x2 square
    tf.keras.layers.MaxPooling2D(2, 2),
# This layer provides the second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#This layer pools the pixels by finding the max pixel in a 2x2 square
    tf.keras.layers.MaxPooling2D(2,2),
# This layer provides the third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#This layer pools the pixels by finding the max pixel in a 2x2 square
    tf.keras.layers.MaxPooling2D(2,2),
#This layer provides the fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#This layer pools the pixels by finding the max pixel in a 2x2 square
    tf.keras.layers.MaxPooling2D(2,2),
#This layer flattens the data to feed it to the DNN
    tf.keras.layers.Flatten(),
#This layer drops out some neurons to reduce overfitting
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(training_generator, epochs=2, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend(loc=0)
plt.figure()

plt.show()
