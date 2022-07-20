import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, LeakyReLU, Conv2DTranspose, Reshape, ReLU
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
from tqdm import trange

# データセット(mnist)の読み込み
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


# データの前処理
train_images = np.asarray(train_images).astype(np.float32)
train_labels = np.asarray(train_labels).astype(np.int32)
test_images = np.asarray(test_images).astype(np.float32)
test_labels = np.asarray(test_labels).astype(np.int32)

# 画像データを0 ~ 1にリスケール
train_images = train_images.reshape(len(train_images), 1, 28, 28) / 255.0
test_images = test_images.reshape(len(test_images), 1, 28, 28) / 255.0


# ハイパーパラメータの設定
batch_size = 128
lam = 0.1
gan_epoch = 200
dim_z = 100         # of the dimension of latent variable
slope = 0.02        # slope of leaky relu
b1 = 0.5            # momentum term of adam
lr1 = 0.001         # initial learning rate for adam
lr2 = 0.0001        # subsequent learning rate for adam


# 学習モデルの定義
class discriminator(Model):

    def __init__(self):
        super().__init__()
        self._conv1 = Conv2D(8, 3, padding='same', strides=1)
        self._conv2 = Conv2D(16, 2, padding='same', strides=2)
        self._conv3 = Conv2D(32, 3, padding='same', strides=2)
        self._conv4 = Conv2D(64, 2, padding='same', strides=2)
        self._conv5 = Conv2D(128, 3, padding='same', strides=2)
        self.dr1 = Dropout()
        self.dr2 = Dropout()
        self.dr3 = Dropout()
        self.dr4 = Dropout()
        self.dr5 = Dropout()
        self.fl = Flatten()
        self._full = Dense(1)

    def forward(self, x):
        self.fx = LeakyReLU(self._conv1(x), alpha=slope)
        h1 = self.dr1(self.fx)
        h2 = self.dr2(LeakyReLU(self._conv2(h1), alpha=slope))
        h3 = self.dr3(LeakyReLU(self._conv3(h2), alpha=slope))
        h4 = self.dr4(LeakyReLU(self._conv4(h3), alpha=slope))
        h5 = self.dr5(LeakyReLU(self._conv5(h4), alpha=slope))
        h6 = self.fl(h5)
        h7 = self._full(h6)
        return h7


class generator(Model):

    def __init__(self):
        super().__init__()
        self._deconv1 =Conv2DTranspose(64, 2, strides=2)
        self._deconv2 =Conv2DTranspose(32, 3, strides=2)
        self._deconv3 =Conv2DTranspose(16, 4, strides=2)
        self._deconv4 =Conv2DTranspose(1, 1, strides=1)
        self._full = Dense(128*3*3, input_dim=dim_z)
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()

    def forward(self, z):
        z = Reshape((-1, 128, 3, 3), input_shape=self._full(z))
        h1 = ReLU(self._deconv1(self.bn1(z)))
        h2 = ReLU(self._deconv2(self.bn2(h1)))
        h3 = ReLU(self._deconv3(self.bn3(h2)))
        h4 = self._deconv5(h3)
        h5 = tf.math.tanh(h4)
        return h5


class dcgan(Model):
    def __init__(self, gen, dis, minimax=False):
        self.gen = gen
        self.dis = dis
        self.minimax = minimax

    def forward(self, x):
        batch = len(x)
        z = np.random.randn(batch*dim_z).reshape((batch, dim_z)).astype(np.float32)
        self.x_gen = self.gen(z)
        self.real_dis = self.dis(self.x_gen)
        self.fake_dis = self.dis(self.x_gen)
        self.prob_real = tf.math.sigmoid(self.real_dis)
        self.prob_fake = tf.math.sigmoid(self.fake_dis)
        self.dis_loss_real = tf.compat.v1.losses.sigmoid_cross_entropy(np.ones(batch).reshape(-1,1), self.real_dis)
        self.dis_loss_fake = tf.compat.v1.losses.sigmoid_cross_entropy(np.zeros(batch).reshape(-1,1), self.fake_dis)
        self.dis_loss = self.dis_loss_real + self.dis_loss_fake
        if self.minimax:
            self.gen_loss = -self.dis_loss
        else:
            self.gen_loss = tf.compat.v1.losses.sigmoid_cross_entropy(np.ones(batch).reshape(-1,1), self.fake_dis)

        return self.dis_loss


# 訓練
dis_opt = Adam(learning_rate=lr1, beta_1=b1)
gen_opt = Adam(learning_rate=lr1, beta_1=b1)

dis = discriminator()
gen = generator()
gan = dcgan(gen, dis)

N = len(x_train)

loss_curve_dis = []
loss_curve_gen = []
acc_curve_real = []
acc_curve_fake = []

for epoch in trange(1, gan_epoch+1):
    perm = np.random.permutation(N)
    total_loss_dis = 0
    total_loss_gen = 0
    total_acc_real = 0
    total_acc_fake = 0

    if epoch <= (gan_epoch - (gan_epoch // 2)):
        dis_opt._lr = lr1 - (lr1 - lr2) * epoch / (gan_epoch - (gan_epoch // 2))
        gen_opt._lr = lr1 - (lr1 - lr2) * epoch / (gan_epoch - (gan_epoch // 2))

    for i in range(N // batch_size):
        index = perm[i*batch_size:(i+1)*batch_size]
        train_batch = x_train[index]
        with gan.train():
            dl = gan(train_batch)
        with gan.gen.prevent_update():
            dl = gan.dis_loss
            dl.grad(detach_graph=False).update(dis_opt)
        with gan.dis.prevent_update():
            gl = gan.gen_loss
            gl.grad().update(gen_opt)
        real_acc = len(np.where(gan.prob_real.as_ndarray() >= 0.5)[0]) / batch_size
        fake_acc = len(np.where(gan.prob_fake.as_ndarray() >= 0.5)[0]) / batch_size
        dis_loss_ = gan.dis_loss.as_ndarray()
        gen_loss_ = gan.gen_loss.as_ndarray()
        total_loss_dis += dis_loss_
        total_loss_gen += gen_loss_
        total_acc_real += real_acc
        total_acc_fake += fake_acc
    loss_curve_dis.append(total_loss_dis/(N/batch_size))
    loss_curve_gen.append(total_loss_gen/(N/batch_size))
    acc_curve_real.append(total_acc_real/(N/batch_size))
    acc_curve_fake.append(total_acc_fake/(N/batch_size))

    if epoch % 10 == 0:
        print("Epoch {} Loss of Dis {:.3f} Loss of Gen {:.3f} Accuracy of Real {:.3f} Accuracy of fake {:.3f}".format(epoch, loss_curve_dis[-1], loss_curve_gen[-1], acc_curve_real[-1], acc_curve_fake[-1]))


# 結果のプロット
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
ax = ax.ravel()
ax[0].plot(loss_curve_dis, linewidth=2, label="dis")
ax[0].plot(loss_curve_gen, linewidth=2, label="gen")]
ax[0].set_title("Learning curve")
ax[0].set_ylabel("loss")
ax[0].set_xlabel("epoch")
ax[0].legend()
ax[0].grid()
ax[0].plot(acc_curve_real, linewidth=2, label="real")
ax[0].plot(acc_curve_fake, linewidth=2, label="fake")]
ax[0].set_title("Accuracy curve")
ax[0].set_ylabel("accuracy")
ax[0].set_xlabel("epoch")
ax[0].legend()
ax[0].grid()

ncols = 10
nrows = 4
z = np.random.randn(ncols*nrows*dim_z).reshape((ncols*nrows, dim_z)).astype(np.float32)
gen_images = gen.gen(z).as_ndarray()
imshow(gen_images)
plt.show()


# 潜在変数の探索
def res_loss(x, z):
    Gz = gan.gen(z)
    abs_sub = abs(x - Gz)
    return tf.math.reduce_sum(abs_sub)

def dis_loss(x, z):
    # compute f(x)
    dl = gan.dis(x)
    fx = gan.dif.fx.as_ndarray()

    # compute f(G(z))
    Gz = gan.gen(z)
    dl = gan.dis(Gz)
    G_fx = gan.dis.fx

    abs_sub = abs(fx - G_fx)
    return tf.math.reduce_sum(abs_sub)

def Loss(x, z):
    return (1-lam)*res_loss(x, z) + lam*dis_loss(x, z)

def numerical_diff(f, x, z):
    with gan.trian():
        loss = f(x, z)
        diff = loss.grad().get(z)
    return np.array(diff)

def grad_descent(f, x, niter=Gamma):
    z_gamma = tf.Variable(np.random.randn(dim_z).reshape((1, dim_z)).astype(np.float32))
    lr = 0.1
    for _ in range(niter):
        z_gamma -= lr * numerical_diff(Loss, x, z_gamma)
    return z_gamma

def Anomaly_score(image_set, nrows=1, ncols=5, figsize=(12.5, 5), normal=True):
    plot_num = nrows * ncols
    _, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize(12, 12*nrows/ncols))
    plt.tight_layout()
    ax = ax.ravel()
    for i in range(plot_num):
        idx = np.random.choice(np.arange(image_set.shape[0]))
        ax[i].imshow(-image_set[idx].reshape(28, 28), cmap='gray')
        x = image_set[idx].reshape((1, 1, 28, 28))
        z_Gamma = grad_descent(Loss, x)
        a_score = Loss(x, z_Gamma)
        if normal:
            ax[i].set_title("Normal: \n"+str(a_score))
        else:
            ax[i].set_title("Anomalous: \n"+str(a_score))
        ax[i].set_xticks([])
        ax[i].set_yticks([])

gan.set_models(inference=True)
Anomaly_score(test_images)

scores = []
plot_num = 1000
for i in range(plot_num):
    idx = np.random.choice(np.arange(test_images.shape[0]))
    x = test_images[idx].reshape((1, 1, 28, 28))
    z_Gamma = grad_descent(Loss, x)
    scores.append(Loss(x, z_Gamma))

scores_sort = sorted(scores, reverse=True)
th = scores_sort[29]
print(th)






