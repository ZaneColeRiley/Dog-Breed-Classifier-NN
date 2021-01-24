# Imports
import tensorflow as tf
import keras
import os
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

IMAGE_DIR = "Images"
DOG_BREED_IMAGE_BASE_DIR = os.path.join(IMAGE_DIR, "images")
DOG_BREED_IMAGE_DIR = os.path.join(DOG_BREED_IMAGE_BASE_DIR, "Images")
TEST_IMAGE_DIR = os.path.join("n02086910-papillon", "n02086910_21.jpg")

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
BATCH_SIZE = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DOG_BREED_IMAGE_DIR,
    validation_split=0.2,
    subset='training',
    seed=325,
    image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DOG_BREED_IMAGE_DIR,
    subset="validation",
    validation_split=0.2,
    seed=352,
    image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DOG_BREED_IMAGE_DIR,
    seed=123,
    image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

train_ds = train_ds.cache().shuffle(10000)
val_ds = val_ds.cache()

normalization_layer = keras.layers.experimental.preprocessing.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

data_augmentation = Sequential([
    keras.layers.experimental.preprocessing.RandomFlip("horizontal",
                                                       input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
    keras.layers.experimental.preprocessing.RandomRotation(0.1),
    keras.layers.experimental.preprocessing.RandomZoom(0.1)
])

num_classes = 120

model = Sequential(name="Dog_Breed_Classifier", layers=[
    data_augmentation,
    keras.layers.experimental.preprocessing.Rescaling(1. / 255),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes)])

model.compile(optimizer=Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 5

checkpoint = ModelCheckpoint("Models/Dog_Classifier.h5", save_best_only=True)


def train_model():
    model.load_weights("Models/Dog_Classifier.h5")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint]
    )


def predict_breed(image):
    model.load_weights("Models/Dog_Classifier.h5")
    img = keras.preprocessing.image.load_img(image,
                                             target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))

    image_array = keras.preprocessing.image.img_to_array(img)
    image_array = tf.expand_dims(image_array, 0)

    predictions = model.predict(image_array, verbose=True)
    score = tf.nn.softmax(predictions[0])

    print("This image most likely belongs to {} with a {:.2f} percent confidence."
          .format(class_names[np.argmax(score)], 100 * np.max(score)))
    return class_names[np.argmax(score)], 100 * np.max(score)


def evaluate_model():
    model.load_weights("Models/Dog_Classifier.h5")
    loss, acc = model.evaluate(test_ds, batch_size=32)
    print("accuracy: {}%".format(100 * acc))


train_model()
# predict_breed("Images/images/Images/n02086910-papillon/n02086910_103.jpg")
# evaluate_model()
