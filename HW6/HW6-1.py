import os
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from io import BytesIO

# Step 1: 設定資料集路徑
dataset_path = "./HW6/dataset"  # 資料集路徑
train_path = dataset_path  # 直接使用資料集的根目錄

# Step 2: 構建 VGG16 模型
def build_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # 固定預訓練層
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')  # 分為 2 類：戴口罩與不戴口罩
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 3: 資料準備與訓練
def train_model(model):
    # 使用 ImageDataGenerator 進行資料增強並拆分資料集
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 20% 資料作為驗證集
    )

    # 訓練資料生成器
    train_generator = train_datagen.flow_from_directory(
        train_path,  # 資料集根目錄
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'  # 使用訓練集資料
    )

    # 驗證資料生成器
    validation_generator = train_datagen.flow_from_directory(
        train_path,  # 資料集根目錄
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'  # 使用驗證集資料
    )

    # 訓練模型
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10
    )
    return history

# Step 4: 處理並預測輸入的圖片
def test_image(image_url, model, class_names):
    # 下載圖片
    response = requests.get(image_url)
    img = image.load_img(BytesIO(response.content), target_size=(224, 224))

    # 預處理圖片
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # 與訓練數據保持一致的預處理方式

    # 預測
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_names[predicted_class]
    
    # 輸出預測結果
    print(f"Predicted class: {predicted_label}")
    return predicted_label

# 主程式
if __name__ == "__main__":
    # 步驟 1: 構建 VGG16 模型
    model = build_vgg16_model()

    # 步驟 2: 訓練模型
    train_model(model)

    # 步驟 3: 輸入圖片 URL 並預測
    image_url = input("請輸入圖片的 URL: ")
    test_image(image_url, model, ['with_mask', 'without_mask'])
