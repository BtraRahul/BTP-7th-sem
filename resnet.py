import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# Step 1: Import Libraries

# Step 2: Prepare Dataset
train_dir = './Training'
validation_dir = './Training'

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')  # Change to 'binary' if it's a binary classification

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')  # Change to 'binary' if it's a binary classification

# Generate label file
class_labels = sorted(train_generator.class_indices.keys())
with open('lb.txt', 'w') as f:
    for label in class_labels:
        f.write(label + '\n')

# Step 3: Load Pre-trained ResNet50 Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model to retain the pre-trained weights
base_model.trainable = False

# Step 4: Add Custom Layers on Top of ResNet50
model = models.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax')  # Output layer with softmax activation
])

# Step 5: Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Change to 'binary_crossentropy' if binary classification
              metrics=['accuracy'])

# Step 6: Train Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)

# Step 7: Evaluate Model
test_loss, test_acc = model.evaluate(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
print('Test accuracy:', test_acc)

# Step 8: Save Model
model.save('resnet_image_classification_model.h5')
