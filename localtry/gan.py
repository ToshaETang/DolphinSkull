import os
import numpy as np
import cv2
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, PReLU
from keras.layers import BatchNormalization, LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import load_model
import matplotlib.pyplot as plt

print("===========================")
print("===========================")
print("start")

plt.switch_backend('agg')

noise_dim = 100
w = 200
h = 200

class GAN(object):
    """ Generative Adversarial Network class """
    def __init__(self, width=w, height=h, channels=1):

        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)
        
        self.optimizer = Adam(learning_rate=0.0002, beta_1=0.6)

        self.G = self.__generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self.D = self.__discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        self.stacked_generator_discriminator = self.__stacked_generator_discriminator()
        self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def __generator(self):
        """ Declare generator """
        model = Sequential()
        model.add(Dense(128, input_shape=(noise_dim,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.width * self.height * self.channels, activation='tanh'))
        model.add(Reshape((self.width, self.height, self.channels)))
        return model

    def __discriminator(self):
        """ Declare discriminator """
        model = Sequential()
        model.add(Flatten(input_shape=self.shape))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model

    def __stacked_generator_discriminator(self):
        self.D.trainable = False
        model = Sequential()
        model.add(self.G)
        model.add(self.D)
        return model
    
    def train(self, X_train, epochs=100000, batch=32, save_interval=100):
        for cnt in range(epochs + 1):
            # train discriminator
            random_index = np.random.randint(0, len(X_train) - batch // 2)
            legit_images = X_train[random_index: random_index + batch // 2]
            gen_noise = np.random.normal(0, 1, (batch // 2, noise_dim))
            synthetic_images = self.G.predict(gen_noise)
            x_combined_batch = np.concatenate((legit_images, synthetic_images))
            y_combined_batch = np.concatenate((np.ones((batch // 2, 1)), np.zeros((batch // 2, 1))))
            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)
            # train generator
            noise = np.random.normal(0, 1, (batch, noise_dim))
            y_mislabeled = np.ones((batch, 1))
            g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabeled)
            #print('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))
            print("epoch:", cnt, "[Discriminator :: d_loss: ", d_loss[0], "], [ Generator :: loss:", g_loss, "]")
            if cnt % save_interval == 0:
                self.plot_images(save2file=True, step=cnt)
                self.save_model(cnt)

    def plot_images(self, save2file=False, samples=16, step=0):
        if not os.path.exists("./images"):
            os.makedirs("./images")
        filename = "./images/dolphin_%d.png" % step
        noise = np.random.normal(0, 1, (samples, noise_dim))
        images = self.G.predict(noise)
        plt.figure(figsize=(4, 4))  # figsize in inches
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            image = images[i, :, :, 0]  # Extract the grayscale channel
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

    def save_model(self, step):
        model_path = "./models"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.G.save(os.path.join(model_path, f"generator_{step}.h5"))
        self.D.save(os.path.join(model_path, f"discriminator_{step}.h5"))

if __name__ == '__main__':
    image_folder_path = r'origin'
    image_files = os.listdir(image_folder_path)
    X_train = []
    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None and not image.size == 0:
            # 如果图片成功读取，并且尺寸不为空，则进行调整大小
            image = cv2.resize(image, (w, h))
            X_train.append(image)
        else:
            print(image_file)
    X_train = np.array(X_train) / 255.0
    X_train = np.expand_dims(X_train, axis=-1)
    gan = GAN()
    gan.train(X_train)



'''
import os
import numpy as np
import cv2
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, PReLU
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import matplotlib.pyplot as plt

print("===========================")
print("===========================")
print("start")

plt.switch_backend('agg')

noise_dim = 100
w = 200
h = 200

class GAN(object):
    """ Generative Adversarial Network class """
    def __init__(self, width=w, height=h, channels=1):

        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)
        
        self.optimizer = Adam(learning_rate=0.0002, beta_1=0.6)
        self.optimizerG = Adam(learning_rate=0.0002, beta_1=0.6)
        self.optimizerD = Adam(learning_rate=0.0003, beta_1=0.6)

        self.G = self.__generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizerG)

        self.D = self.__discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizerD, metrics=['accuracy'])

        self.stacked_generator_discriminator = self.__stacked_generator_discriminator()

        self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def __generator(self):
        """ Declare generator """
        model = Sequential()
        model.add(Dense(128, input_shape=(noise_dim,)))  # noise_dim
        model.add(LeakyReLU(alpha=0.2))  # LeakyReLU(alpha=0.2)  # PReLU()
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.width * self.height * self.channels, activation='tanh'))
        model.add(Reshape((self.width, self.height, self.channels)))
        return model

    def __discriminator(self):
        """ Declare discriminator """
        model = Sequential()
        model.add(Flatten(input_shape=self.shape))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model

    def __stacked_generator_discriminator(self):
        self.D.trainable = False
        model = Sequential()
        model.add(self.G)
        model.add(self.D)
        return model
    
    def train(self, X_train, epochs=100000, batch=32, save_interval=1000):
        for cnt in range(epochs + 1):
            # train discriminator
            random_index = np.random.randint(0, len(X_train) - np.int64(batch / 2))
            legit_images = X_train[random_index: random_index + np.int64(batch / 2)]
            gen_noise = np.random.normal(0, 1, (np.int64(batch / 2), noise_dim))  # noise_dim
            syntetic_images = self.G.predict(gen_noise)
            x_combined_batch = np.concatenate((legit_images, syntetic_images))
            y_combined_batch = np.concatenate((np.ones((np.int64(batch / 2), 1)), np.zeros((np.int64(batch / 2), 1))))
            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)
            # train generator
            noise = np.random.normal(0, 1, (batch, noise_dim))  # noise_dim
            y_mislabled = np.ones((batch, 1))
            g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled)
            print('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))
            if cnt % save_interval == 0:
                self.plot_images(save2file=True, step=cnt)
                self.save_model(cnt)

    def plot_images(self, save2file=False, samples=16, step=0):
        if not os.path.exists("./images"):
            os.makedirs("./images")
        filename = "./images/dolphin_%d.png" % step
        noise = np.random.normal(0, 1, (samples, noise_dim))  # noise_dim
        images = self.G.predict(noise)
        plt.figure(figsize=(w, h))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            image = images[i, :, :, 0]  # 取出灰階通道
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

    def save_model(self, step):
        model_path = "./models"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.G.save(os.path.join(model_path, f"generator_{step}.h5"))
        self.D.save(os.path.join(model_path, f"discriminator_{step}.h5"))

if __name__ == '__main__':
    image_folder_path = r'origin'
    image_files = os.listdir(image_folder_path)
    X_train = []
    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None and not image.size == 0:
            # 如果圖片成功讀取，並且尺寸不為空，則進行調整大小
            image = cv2.resize(image, (w, h))
            X_train.append(image)
        else:
            print(image_file)
    X_train = np.array(X_train) / 255.0
    X_train = np.expand_dims(X_train, axis=-1)
    gan = GAN()
    gan.train(X_train)
'''