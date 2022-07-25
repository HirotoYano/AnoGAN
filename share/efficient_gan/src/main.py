import os
import datetime
import numpy as np

from tensorflow.keras.datasets import mnist

from efficient_gan import EfficientGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d-%H-%M")
output_dir = "../output/" + current_time
os.makedirs(output_dir, exist_ok=True)

label: int = 1

# data load
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"X_train : {X_train.shape}")
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
print(f"x_train : {x_train.shape}")

model = EfficientGAN(output_dir=output_dir)

model.fit(x_train, test=(x_test, y_train))
proba = model.predict(x_test)
# print(f"proba : {proba.shape}")
# print(f"proba : {proba}")

# proba_reshape = proba.reshape([28, 28])
# print(f"proba_reshape : {proba_reshape.shape}")
