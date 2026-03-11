import tensorflow as tf

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "training",
    image_size=(224,224),
    batch_size=32
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    "testing",
    image_size=(224,224),
    batch_size=32
)

print(train_data.class_names)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(16,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(train_data, epochs=5)

model.evaluate(test_data)

model.save("brain_tumor_model.h5")

print("Model saved successfully!")

loss, accuracy = model.evaluate(test_data)

print("Test Accuracy:", accuracy)
