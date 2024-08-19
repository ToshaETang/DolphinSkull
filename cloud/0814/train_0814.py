
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import layers
import time

print("------------------------")
print("START")

# 自定義參數
image_folder_path = r'origin'
w, h = 100, 100  # 修改圖像尺寸
BUFFER_SIZE = 1000
BATCH_SIZE = 256
EPOCHS = 100001  # 可以設定為需要的訓練回數
noise_dim = 100
num_examples_to_generate = 16

# 讀取本機圖像文件
image_files = os.listdir(image_folder_path)
X_train = []
for image_file in image_files:
    image_path = os.path.join(image_folder_path, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is not None and not image.size == 0:
        image = cv2.resize(image, (w, h))
        X_train.append(image)
    else:
        print(image_file)

X_train = np.array(X_train) / 255.0 * 2 - 1  # Normalize to [-1, 1]
X_train = np.expand_dims(X_train, axis=-1)
train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(25*25*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((25, 25, 256)))
    assert model.output_shape == (None, 25, 25, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 50, 50, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 100, 100, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 100, 100, 1)

    return model

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')




from tensorflow.keras.regularizers import l2

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[100, 100, 1],
                            kernel_regularizer=l2(0.01)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=l2(0.01)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=l2(0.01)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=l2(0.01)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model







#def make_discriminator_model():
#    model = tf.keras.Sequential()
#    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[100, 100, 1]))
#    model.add(layers.LeakyReLU())
#    model.add(layers.Dropout(0.3))

#    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
#    model.add(layers.LeakyReLU())
#    model.add(layers.Dropout(0.3))

#    model.add(layers.Flatten())
#    model.add(layers.Dense(1))

#    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "0802_ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# 加載檢查點（如果存在）
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
initial_epoch = 0

if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print("檢查點加載成功：", latest_checkpoint)
    # 加載儲存的訓練輪數
    epoch_file = os.path.join(checkpoint_dir, "epoch.txt")
    if os.path.exists(epoch_file):
        with open(epoch_file, "r") as f:
            initial_epoch = int(f.read())
else:
    print("沒有找到檢查點，從頭開始訓練")

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs, initial_epoch):
    for epoch in range(initial_epoch, epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            # 儲存當前的訓練輪數
            epoch_file = os.path.join(checkpoint_dir, "epoch.txt")
            with open(epoch_file, "w") as f:
                f.write(str(epoch + 1))

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    if epoch % 10 == 0:
        plt.savefig(os.path.join('image_at_epoch_0802', 'epoch_{:04d}.png'.format(epoch)))

train(train_dataset, EPOCHS, initial_epoch)
