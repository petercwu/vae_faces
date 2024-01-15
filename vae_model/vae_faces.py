import math
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, Callback
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return tf.random.normal(tf.shape(log_var)) * tf.exp(log_var / 2) + mean


def plot_reconstructions(model, images, n_images=25, rows=5, cols=5):
    reconstructions = np.clip(model.predict(images[:n_images]), 0, 1)
    fig = plt.figure(figsize=(cols * 2, rows))
    for image_index in range(n_images):
        plt.subplot(rows, cols * 2, 1 + 2 * image_index)
        plt.imshow(images[image_index])
        plt.axis("off")
        plt.subplot(rows, cols * 2, 2 + 2 * image_index)
        plt.imshow(reconstructions[image_index])
        plt.axis("off")
        plt.subplots_adjust(wspace=0.1)


def kl_loss(z_mean, z_log_var):
    return -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)

# Saves the model only if its loss has improved
class SaveBestModelsCallback(Callback):
    def __init__(self, encoder_model, decoder_model):
        super(SaveBestModelsCallback, self).__init__()
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        avg_val_loss = sum(val_loss) / len(val_loss)
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.encoder_model.save(f"best_vae_encoder_faces_model_epoch_{epoch}")
            self.decoder_model.save(f"best_vae_decoder_faces_model_epoch_{epoch}")

# Hyperparameters
tf.random.set_seed(42)
folder_path = "INSERT FOLDER PATH HERE"
latent_dim = 250
batchSize = 128
epochs = 5
kl_weight = 0.0001
img_height = 216
img_width = 176
LR = 0.0001
inputShape = (None, img_height, img_width, 3)

# Encoder model
encoderInput = tf.keras.layers.Input(shape=inputShape[1:], name="encoder_input")
x = tf.keras.layers.Conv2D(filters=32, strides=2, kernel_size=3, padding="same")(encoderInput)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Dropout(rate=0.25)(x)
x = tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Dropout(rate=0.25)(x)
x = tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Dropout(rate=0.25)(x)
x = tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Dropout(rate=0.25)(x)
x = tf.keras.layers.Flatten()(x)
z_mean = tf.keras.layers.Dense(latent_dim)(x)
z_log_var = tf.keras.layers.Dense(latent_dim)(x)

# Sample the latent vector z using the reparameterization trick
z = Sampling()([z_mean, z_log_var])
vae_encoder = tf.keras.Model(inputs=[encoderInput], outputs=[z], name="vae_encoder")

# Decoder model
decoderInput = tf.keras.layers.Input(shape=[latent_dim], name="decoder_input")
x = tf.keras.layers.Dense(27*22*64)(decoderInput)
x = tf.keras.layers.Reshape((27, 22, 64))(x)
x = tf.keras.layers.Conv2DTranspose(filters=64, strides=2, kernel_size=3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Dropout(rate=0.25)(x)
x = tf.keras.layers.Conv2DTranspose(filters=32, strides=2, kernel_size=3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Dropout(rate=0.25)(x)
x = tf.keras.layers.Conv2DTranspose(filters=32, strides=2, kernel_size=3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Dropout(rate=0.25)(x)
x = tf.keras.layers.Conv2DTranspose(filters=3, strides=1, kernel_size=3, padding="same")(x)
x = tf.keras.layers.Activation('sigmoid')(x)
vae_decoder = tf.keras.Model(inputs=[decoderInput], outputs=[x], name="vae_decoder")

z = vae_encoder(encoderInput)
reconstructions = vae_decoder(z)
vae = tf.keras.Model(inputs=[encoderInput], outputs=[reconstructions])
print(vae_encoder.summary())
print(vae_decoder.summary())

# Add the KL loss to the VAE model
kl_loss_value = kl_loss(z_mean, z_log_var)
vae.add_loss((tf.reduce_mean(kl_loss_value) / 1024) * kl_weight)

# Compile the models
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
vae_encoder.compile(loss="mse", optimizer=optimizer)
vae_decoder.compile(loss="mse", optimizer=optimizer)
vae.compile(loss="mse", optimizer=optimizer)

# Using data generators, process and load the images batch by batch
data_gen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

train_generator = data_gen.flow_from_directory(
    folder_path,
    target_size=(img_height, img_width), # Input shape of (218, 178, 3)
    batch_size=batchSize,
    class_mode=None
)
# print("Number of samples in train_generator: ", train_generator.samples)

steps_per_epoch = train_generator.samples // batchSize
print("Steps per epoch: ", steps_per_epoch)
total_images = 0
losses = []
reconstruct_img = next(train_generator)

# Create instances of earlystopping and save_best_model classes
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=True, restore_best_weights=True)
save_best_models_callback = SaveBestModelsCallback(vae_encoder, vae_decoder)

# Train the model
for epoch in range(epochs):
    for step in range(steps_per_epoch):
        print(f"Epoch {epoch + 1}/{epochs} Batch {step + 1}/{steps_per_epoch}")
        batch_img = next(train_generator)
        history = vae.fit(batch_img, batch_img, validation_data=(batch_img, batch_img),
                                    callbacks=[early_stopping])
    save_best_models_callback.on_epoch_end(epoch, history.history)

    # Plots the losses
    losses.append(history.history)
    plt.figure(figsize=(8, 6))
    plt.plot([loss['loss'][0] for loss in losses], label='Reconstruction and KL Loss', marker='o')
    plt.plot([loss['val_loss'][0] for loss in losses], label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('VAE Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig("vae_faces_training_losses.png")
    plt.close()

    # Plots the reconstructions
    plot_reconstructions(vae, images=reconstruct_img)
    plt.savefig(f"vae_faces_reconstruction_plot_epoch_{epoch + 1}.png")
    plt.close()

# Save the encoder model
encoder_model_name = "vae_faces_encoder_model"
encoder_version = "0001"
encoder_model_path = Path(encoder_model_name) / encoder_version
vae_encoder.save(encoder_model_path, save_format="tf")

# Save the decoder model
decoder_model_name = "vae_faces_decoder_model"
decoder_version = "0001"
decoder_model_path = Path(decoder_model_name) / decoder_version
vae_decoder.save(decoder_model_path, save_format="tf")