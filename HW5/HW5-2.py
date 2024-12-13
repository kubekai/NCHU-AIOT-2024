import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# 1. 資料載入與預處理
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test_labels = y_test.copy()  # 儲存原始標籤以供顯示
y_test = to_categorical(y_test, 10)

# 2. 模型定義
def build_dnn_model():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. 訓練與評估
def train_and_evaluate(model, model_name):
    print(f"\nTraining {model_name}...")
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=2)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"{model_name} Test Accuracy: {accuracy:.4f}")
    return model, accuracy

# 4. 可視化辨識結果
def visualize_predictions(model, model_name, x_test, y_test_labels):
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)

    # 隨機選擇 10 筆測試資料
    indices = np.random.choice(len(x_test), 10, replace=False)
    samples = x_test[indices]
    true_labels = y_test_labels[indices]
    pred_labels = predicted_labels[indices]

    # 顯示圖片與預測結果
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(samples[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {true_labels[i]}\nPred: {pred_labels[i]}")
        plt.axis('off')
    plt.suptitle(f"{model_name} Prediction Results")
    plt.tight_layout()
    plt.show()

# 5. 執行模型並比較
dnn_model = build_dnn_model()
cnn_model = build_cnn_model()

dnn_model, dnn_accuracy = train_and_evaluate(dnn_model, "DNN")
cnn_model, cnn_accuracy = train_and_evaluate(cnn_model, "CNN")

# 6. 顯示模型辨識結果
print("\nVisualizing DNN Predictions:")
visualize_predictions(dnn_model, "DNN", x_test, y_test_labels)

print("\nVisualizing CNN Predictions:")
visualize_predictions(cnn_model, "CNN", x_test, y_test_labels)

# 7. 最終比較
print("\nModel Comparison:")
print(f"DNN Accuracy: {dnn_accuracy:.4f}")
print(f"CNN Accuracy: {cnn_accuracy:.4f}")
