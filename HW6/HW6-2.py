import os
import zipfile
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np

# 下載與解壓數據集
def download_and_extract_data():
    import requests

    url1 = "https://storage.googleapis.com/learning-datasets/rps.zip"
    url2 = "https://storage.googleapis.com/learning-datasets/rps-test-set.zip"

    os.makedirs('/tmp', exist_ok=True)

    with open('/tmp/rps.zip', 'wb') as f:
        f.write(requests.get(url1).content)
    with open('/tmp/rps-test-set.zip', 'wb') as f:
        f.write(requests.get(url2).content)

    with zipfile.ZipFile('/tmp/rps.zip', 'r') as zip_ref:
        zip_ref.extractall('/tmp/')
    with zipfile.ZipFile('/tmp/rps-test-set.zip', 'r') as zip_ref:
        zip_ref.extractall('/tmp/')

# 初始化數據集
def prepare_data():
    download_and_extract_data()

    rock_dir = os.path.join('/tmp/rps/rock')
    paper_dir = os.path.join('/tmp/rps/paper')
    scissors_dir = os.path.join('/tmp/rps/scissors')

    print('total training rock images:', len(os.listdir(rock_dir)))
    print('total training paper images:', len(os.listdir(paper_dir)))
    print('total training scissors images:', len(os.listdir(scissors_dir)))

# 建立數據生成器
def create_generators():
    TRAINING_DIR = "/tmp/rps/"
    VALIDATION_DIR = "/tmp/rps-test-set/"

    training_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        batch_size=126,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150, 150),
        batch_size=126,
        class_mode='categorical')

    return train_generator, validation_generator

# 建立 CNN 模型
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

# 訓練模型
def train_model(model, train_generator, validation_generator):
    history = model.fit(
        train_generator,
        epochs=25,
        steps_per_epoch=20,
        validation_data=validation_generator,
        validation_steps=3,
        verbose=1)
    model.save("rps.h5")
    return history

# 畫出結果
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    plt.show()

def test_model(model_path, img_path):
    model = tf.keras.models.load_model(model_path)
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    print("Prediction:", prediction)

if __name__ == "__main__":
    prepare_data()
    train_generator, validation_generator = create_generators()
    model = build_model()
    history = train_model(model, train_generator, validation_generator)
    plot_history(history)
    test_model("rps.h5", "/tmp/rps/rock/rock01-000.png")
