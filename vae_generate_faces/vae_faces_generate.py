import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np

decoder_model_path = "INSERT DECODER PATH HERE"

loaded_decoder = keras.models.load_model(decoder_model_path)
# Set the number of faces you want to generate
num_faces_to_generate = 10

# Generate random latent vectors
random_latent_vectors = np.random.normal(size=(num_faces_to_generate, 250))
print(random_latent_vectors.shape)
# Generate new faces using the decoder
generated_faces = loaded_decoder.predict(random_latent_vectors)

# Clip the pixel values to [0, 1] range
generated_faces = np.clip(generated_faces, 0, 1)

# Display the generated faces
plt.figure(figsize=(12, 6))
for i in range(num_faces_to_generate):
    plt.subplot(2, 5, i + 1)
    plt.imshow(generated_faces[i])
    plt.axis('off')
plt.show()

