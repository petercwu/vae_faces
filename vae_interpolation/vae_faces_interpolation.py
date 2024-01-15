import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np
import cv2

# Load the models
encoder_model_path = "INSERT ENCODER PATH HERE"
decoder_model_path = "INSERT DECODER PATH HERE"
loaded_encoder = keras.models.load_model(encoder_model_path)
loaded_decoder = keras.models.load_model(decoder_model_path)

# Load the two images
image1_path = "INSERT IMAGE PATH HERE"
image2_path = "INSERT IMAGE PATH HERE"
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)
# Convert from BGR to RGB
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
# Resize to the input shape of the encoder
image1_resized = cv2.resize(image1, (176, 216))
image2_resized = cv2.resize(image2, (176, 216))
# Rescale the images to [0, 1]
image1_data_rescaled = image1_resized / 255.0
image2_data_rescaled = image2_resized / 255.0

# Encode the images to obtain their corresponding latent vectors
latent_vector1 = loaded_encoder.predict(np.expand_dims(image1_data_rescaled, axis=0))
latent_vector2 = loaded_encoder.predict(np.expand_dims(image2_data_rescaled, axis=0))

num_steps = 10
interpolated_latents = []
for i in range(num_steps + 1):
    alpha = i / num_steps
    interpolated_latent = (1 - alpha) * latent_vector1 + alpha * latent_vector2
    interpolated_latents.append(interpolated_latent)

interpolated_images = []
for latent in interpolated_latents:
    interpolated_image = loaded_decoder(latent).numpy().squeeze()
    interpolated_images.append(interpolated_image)

# Display the original images and the interpolated images
plt.figure(figsize=(12, 6))
plt.subplot(1, num_steps + 2, 1)
plt.imshow(image1)
plt.title('Image 1')
plt.xticks([])
plt.yticks([])

for i, image in enumerate(interpolated_images):
    plt.subplot(1, num_steps + 2, i + 2)
    plt.imshow(image)
    plt.title(f'Step {i+1}')
    plt.xticks([])
    plt.yticks([])

plt.subplot(1, num_steps + 2, num_steps + 2)
plt.imshow(image2)
plt.title('Image 2')
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.savefig("faces_image_interpol.png")
plt.show()
