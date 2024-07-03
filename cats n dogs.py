import tensorflow as tf
from keras import layers, models, datasets, losses
import matplotlib.pyplot as plt
import sys, io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_path = tf.keras.utils.get_file('cats_and_dogs.zip', origin=url, extract=True)
base_dir = os.path.join(os.path.dirname(zip_path), 'cats_and_dogs_filtered')

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

num_train_samples = sum([len(files) for r, d, files in os.walk(train_dir)])
num_val_samples = sum([len(files) for r, d, files in os.walk(validation_dir)])

steps_per_epoch = num_train_samples // 25
validation_steps = num_val_samples // 25

gen_data_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1.0 / 255.0,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True
)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
train_generator = gen_data_train.flow_from_directory(
    train_dir,
    target_size =(150,150),
    batch_size = 20,
    class_mode = "binary"
)
valid_generator = valid_datagen.flow_from_directory(
    validation_dir,
    target_size =(150,150),
    batch_size = 20,
    class_mode = "binary"
)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=losses.BinaryCrossentropy, metrics=['accuracy'])
model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=25, validation_data=valid_generator, validation_steps=validation_steps)

valid_loss, valid_accuracy = model.evaluate(valid_generator, steps=50)
print(f'The model was {valid_accuracy}% accurate')

