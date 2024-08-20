import tensorflow as tf
import numpy as np
import ollama

def classify_image(image_path):
    # Load the image
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # Load a pre-trained model (e.g., MobileNetV2)
    model = tf.keras.applications.MobileNetV2(weights='imagenet')

    # Preprocess the image for the model
    processed_image = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

    # Make predictions
    predictions = model.predict(processed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)

    # Return the description of the image
    return decoded_predictions[0][0][1]  # Returning the top prediction

def generate_description(description):
    # Initialize Ollama client
    client = ollama.Client()

    # Generate a response using the specified model
    response = client.generate(model="llama3", prompt=description)
    return response

def main():
    image_path = 'หิวข้าว.png'  # Replace with the path to your image
    description = classify_image(image_path)
    response = generate_description(description)
    print(response)

if __name__ == "__main__":
    main()
