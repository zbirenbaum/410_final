import tensorflow as tf
from pixel_shuffler import PixelShuffler

data_dir = "./data/images_original"
size = (432, 288)
batch_size = 32
train_ds = tf.keras.utils.image_dataset_from_directory(
  directory = data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=size,
  batch_size=1)
val_ds = tf.keras.utils.image_dataset_from_directory(
  directory = data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=size,
  batch_size=32)
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 10 

#Stable at 63-65
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(16, 12, strides=(1,1), padding="same", activation='relu'),
  tf.keras.layers.MaxPooling2D(3),
  tf.keras.layers.Conv2D(16, 8, strides=(1,1), padding="same", activation='relu'),
  tf.keras.layers.MaxPooling2D(2),
  # tf.keras.layers.Conv2D(32, 2, strides=(1,1), padding="same", activation='relu'),
  # tf.keras.layers.MaxPooling2D(2),
  # tf.keras.layers.Conv2D(32, 2, strides=(1,1), padding="same", activation='relu'),
  # tf.keras.layers.MaxPooling2D(2),
  # tf.keras.layers.Conv2D(32, 2, strides=(1,1), padding="same", activation='relu'),
  # tf.keras.layers.MaxPooling2D(2),
  # tf.keras.layers.Conv2D(64, 2, strides=(1,1), padding="same", activation='relu'),
  # tf.keras.layers.MaxPooling2D(2),
  tf.keras.layers.Conv2D(256, 2, strides=(1,1), padding="same", activation='relu'),
  tf.keras.layers.Dropout(.3),
  tf.keras.layers.Conv2D(128, 2, strides=(1,1), padding="same", activation='relu'),
  tf.keras.layers.Dropout(.3),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

#Benchmark

# model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.compile(
  optimizer=tf.keras.optimizers.Adamax(),
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=1000)


#stable at 70-72 after ~400 epoch
# model = tf.keras.Sequential([
#   tf.keras.layers.Rescaling(1./255),
#   tf.keras.layers.Conv2D(32, 8, strides=(1,1), padding="same", activation='relu'),
#   tf.keras.layers.MaxPooling2D(4),
#   tf.keras.layers.Conv2D(32, 8, strides=(1,1), padding="same", activation='relu'),
#   tf.keras.layers.MaxPooling2D(4),
#   tf.keras.layers.Conv2D(32, 4, strides=(1,1), padding="same", activation='relu'),
#   tf.keras.layers.MaxPooling2D(3),
#   tf.keras.layers.Conv2D(64, 3, strides=(1,1), padding="same", activation='relu'),
#   tf.keras.layers.MaxPooling2D(2),
#   tf.keras.layers.Dropout(.3),
#   tf.keras.layers.Conv2D(128, 2, strides=(1,1), padding="same", activation='relu'),
#   tf.keras.layers.Dropout(.3),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(256, activation='relu'),
#   tf.keras.layers.Dense(num_classes)
# ])
#Can hit about 64
# model = tf.keras.Sequential([
#   tf.keras.layers.Rescaling(1./255),
#   tf.keras.layers.Conv2D(16, 4, activation='relu'),
#   tf.keras.layers.Dropout(.3),
#   tf.keras.layers.MaxPooling2D(4),
#   tf.keras.layers.Conv2D(32, 4, activation='relu'),
#   tf.keras.layers.Dropout(.3),
#   tf.keras.layers.MaxPooling2D(4),
#   tf.keras.layers.Conv2D(32, 2, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Dropout(.3),
#   tf.keras.layers.Conv2D(64, 3, activation='relu'),
#   tf.keras.layers.Dropout(.3),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(256, activation='relu'),
#   tf.keras.layers.Dense(num_classes)
# ])
# model = tf.keras.Sequential([
#     tf.keras.layers.Rescaling(1./255),
#     tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
#     tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.25),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(num_classes, activation='softmax'),
# ])
