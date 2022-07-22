import numpy as np

from tensorflow.keras.datasets import mnist

from efficient_gan import EfficientGAN

label: int = 1

# data load
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
# 時系列データ化
x_train = np.array(
        [
            (x.astype('float32') / 255.0).flatten()
            for x in X_train
        ])
x_test = x_train.copy()
# X_train = (X_train.astype(np.float32) - 127.5) / 127.5

# X_train = X_train[y_train == label]
# X_test = X_test[y_test == label]

# print(f"X_train shape:{X_train.shape}\n")
# print(x_train.shape)

model = EfficientGAN()

model.fit(x_train, test=(x_test, y_train))
proba = model.predict(x_test)
# print(f"proba : {proba.shape}")
