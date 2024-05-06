import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

# Load the MobileNetV2 model pre-trained on ImageNet
model = MobileNetV2(weights='imagenet', include_top=True)

def predict_species(image_path):
    # Load and preprocess the input image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array)

    # Use the pre-trained model to predict the top-3 species labels
    predictions = model.predict(processed_img)
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get top-3 predictions
    species_predictions = [(label, prob) for (_, label, prob) in decoded_predictions]

    return species_predictions

# Example usage:
image_path = 'C:/Users/ashme/Downloads/images/Lion.jpg'
predictions = predict_species(image_path)
print("Predicted species:")
for label, prob in predictions:
    print(f"{label}: {prob:.4f}")

# Plot the input image
plt.figure()
img = image.load_img(image_path)
plt.imshow(img)
plt.title("Input Image")
plt.axis('off')
plt.show()
