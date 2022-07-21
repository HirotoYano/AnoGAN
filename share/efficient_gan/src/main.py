import numpy as np

from tensorflow.keras.datasets import mnist

from efficient_gan import EfficientGAN

label: int = 1

# data load
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = (X_train.astype(np.float32) - 127.5) / 127.5

X_train = X_train[y_train == label]
X_test = X_test[y_test == label]

# print(f"X_train shape:{X_train.shape}\n")
print(type(X_train))

# model = EfficientGAN()

# model.fit(X_train)
