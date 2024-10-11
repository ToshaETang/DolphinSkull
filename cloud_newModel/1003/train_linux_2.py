import tensorflow as tf
import os
import time
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#-------------------------------------------
BATCH_SIZE = 64
NOISE_DIM = 50
IMG_SIZE = 512

SAVE_NUM = 1
TRAIN_NUM = 11
#-------------------------------------------
# Creating the models
list_ds = tf.data.Dataset.list_files('./Dolphin_skulls_PNG/*.png')
cat_ds = list_ds.map(lambda x: tf.image.decode_jpeg(tf.io.read_file(x)))

#-------------------------------------------
def configure_for_performance(ds):
    ds = ds.map(lambda x: tf.image.resize(x, (IMG_SIZE, IMG_SIZE))/255)
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = cat_ds
train_ds = configure_for_performance(train_ds)

#-------------------------------------------
# Creating the models
FILTER_COUNT = 16
IN_LAYER_COUNT = 6

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(
        (IMG_SIZE//(2**IN_LAYER_COUNT))*(IMG_SIZE//(2**IN_LAYER_COUNT))*NOISE_DIM, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((IMG_SIZE//(2**IN_LAYER_COUNT),
              IMG_SIZE//(2**IN_LAYER_COUNT), NOISE_DIM)))

    for i in range(IN_LAYER_COUNT):
        model.add(layers.Conv2DTranspose(FILTER_COUNT*2**(IN_LAYER_COUNT-i-1),
                  (6, 6), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (6, 6), strides=(1, 1),
                                     padding='same', use_bias=False, activation='sigmoid'))
    assert model.output_shape == (None, IMG_SIZE, IMG_SIZE, 3)

    return model

#-------------------------------------------
generator = make_generator_model()
noise = tf.random.normal([1, NOISE_DIM])
generated_image = generator(noise, training=False)
generator.summary()

#-------------------------------------------
DISC_FILTER_COUNT = FILTER_COUNT
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(DISC_FILTER_COUNT, (6, 6), strides=(2, 2), padding='same',
                                     input_shape=[IMG_SIZE, IMG_SIZE, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    for i in range(1, IN_LAYER_COUNT):
        model.add(layers.Conv2D(DISC_FILTER_COUNT*(2**i), (6, 6), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation='sigmoid'))

    return model

#-------------------------------------------
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
discriminator.summary()

print (decision)

#-------------------------------------------
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy()

#-------------------------------------------
# Discriminator loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

#-------------------------------------------
# Generator loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

#-------------------------------------------
generator_optimizer = tf.keras.optimizers.Adam(0.001)
discriminator_optimizer = tf.keras.optimizers.Adam(0.002)

#-------------------------------------------
# Ensure directories exist
checkpoint_dir = './models_2'
generated_dir = './generated_2'
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

#-------------------------------------------
# Checkpoint saving
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

#-------------------------------------------
num_examples_to_generate = 1
seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])

#-------------------------------------------
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(dataset, epochs):
    gen_losses = []
    disc_losses = []

    for epoch in range(epochs):
        start = time.time()
        gen_loss = disc_loss = batch_count = 0

        for image_batch in dataset:
            step_gen_loss, step_disc_loss = train_step(image_batch)
            gen_loss += step_gen_loss
            disc_loss += step_disc_loss
            batch_count += 1

        gen_losses.append(gen_loss / batch_count)
        disc_losses.append(disc_loss / batch_count)

        # Produce and save images for each epoch
        generate_and_save_images(generator, epoch + 1, seed, epoch % SAVE_NUM == 0)

        # Save the model every 20 epochs
        if (epoch + 1) % SAVE_NUM == 0:
            manager.save()

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Save the loss plot after training
    save_loss_plot(gen_losses, disc_losses)

#-------------------------------------------
def generate_and_save_images(model, epoch, test_input, save=False):
    predictions = model(test_input, training=False)

    if save:
        for i in range(predictions.shape[0]):
            img = predictions[i, :, :, :]
            img_path = os.path.join(generated_dir, 'image_at_epoch_{:04d}_{}.png'.format(epoch, i))
            tf.keras.preprocessing.image.save_img(img_path, img)

#-------------------------------------------
def save_loss_plot(gen_losses, disc_losses):
    plt.figure()
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator and Discriminator Losses Over Time')
    plt.savefig(os.path.join(generated_dir, 'loss_plot.png'))
    plt.close()

#-------------------------------------------
train(train_ds, TRAIN_NUM)

#-------------------------------------------
# Restore the latest checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
