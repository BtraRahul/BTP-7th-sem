import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet101
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# Paths to image directoriesz
glioma_tumor_dir = 'Training/glioma_tumor'
meningioma_tumor_dir = 'Training/meningioma_tumor'
normal_dir = 'Training/no_tumor'
pituitary_tumor_dir = 'Training/pituitary_tumor'

# Paths to training and validation directories
# # working_dir = '/home/batraji/Projects/folder'
# train_dir = "Training/"
# val_dir = "Training/"

# Paths to image directories (local dataset)

# Paths to training and validation directories (output folders)
working_dir = 'brain-tumor-dataset/working'
train_dir = os.path.join(working_dir, 'train')
val_dir = os.path.join(working_dir, 'val')


def display_sample_images(image_dir, title, num_samples=5):
    plt.figure(figsize=(12, 4))
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)[:num_samples]]
    for i, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()

display_sample_images(glioma_tumor_dir, 'Glioma Tumor')
display_sample_images(meningioma_tumor_dir, 'Meningioma Tumor')
display_sample_images(normal_dir, 'Normal')
display_sample_images(pituitary_tumor_dir, 'Pituitary Tumor')

def get_image_count(image_dir):
    return len([name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))])

glioma_count = get_image_count(glioma_tumor_dir)
meningioma_count = get_image_count(meningioma_tumor_dir)
normal_count = get_image_count(normal_dir)
pituitary_count = get_image_count(pituitary_tumor_dir)

plt.figure(figsize=(10, 5))
plt.bar(['Glioma', 'Meningioma', 'Normal', 'Pituitary'], [glioma_count, meningioma_count, normal_count, pituitary_count])
plt.ylabel('Number of Images')
plt.title('Dataset Distribution')
plt.show()

def get_image_paths(directory_path):
    image_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def organize_data(image_paths, label, train_dir, val_dir, test_size=0.2):
    train_paths, val_paths = train_test_split(image_paths, test_size=test_size, random_state=42)
    def copy_images(paths, dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        for path in paths:
            shutil.copy(path, os.path.join(dest_dir, os.path.basename(path)))
    train_label_dir = os.path.join(train_dir, label)
    val_label_dir = os.path.join(val_dir, label)
    copy_images(train_paths, train_label_dir)
    copy_images(val_paths, val_label_dir)
    return len(train_paths), len(val_paths)
# Get image paths
glioma_image_paths = get_image_paths(glioma_tumor_dir)
meningioma_image_paths = get_image_paths(meningioma_tumor_dir)
normal_image_paths = get_image_paths(normal_dir)
pituitary_image_paths = get_image_paths(pituitary_tumor_dir)

# Organize data
num_train_glioma, num_val_glioma = organize_data(glioma_image_paths, 'glioma_tumor', train_dir, val_dir)
num_train_meningioma, num_val_meningioma = organize_data(meningioma_image_paths, 'meningioma_tumor', train_dir, val_dir)
num_train_normal, num_val_normal = organize_data(normal_image_paths, 'normal', train_dir, val_dir)
num_train_pituitary, num_val_pituitary = organize_data(pituitary_image_paths, 'pituitary_tumor', train_dir, val_dir)

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet.preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


model = models.Sequential()
model.add(ResNet101(include_top=False, weights='imagenet', pooling='avg'))
model.add(layers.Dense(4, activation='softmax'))

for layer in model.layers[0].layers:
    layer.trainable = False
    
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the trained model
model_save_path = "resnet101_trained_model.h5"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")


# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
print("Validation Accuracy: {}".format(val_accuracy))

# Plot training & validation accuracy values
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()