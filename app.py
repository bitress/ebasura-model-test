import numpy as np
import tensorflow as tf
import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details of the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Preprocessing function
def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # Convert the image to grayscale if required
    input_shape = input_details[0]['shape']
    if input_shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to match the model input size
    height, width = int(input_shape[1]), int(input_shape[2])
    image_resized = cv2.resize(image, (width, height))

    # Normalize the image
    image_normalized = image_resized / 255.0

    # Expand dimensions to match the input shape
    if input_shape[-1] == 1:
        input_tensor = np.expand_dims(image_normalized, axis=-1)  # Add channel dimension for grayscale
    else:
        input_tensor = image_normalized

    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)  # Add batch dimension
    return input_tensor, image


# Function to run inference and display results
def recognize_image(image_path):
    try:
        # Preprocess the image
        input_tensor, original_image = preprocess_image(image_path)

        # Set input tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = np.argmax(output_data[0])

        # Load labels
        labels = []
        with open('labels.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        predicted_label = labels[predicted_index]
        confidence = output_data[0][predicted_index]

        # Display prediction result
        result_label.config(text=f"Predicted Label: {predicted_label} ({confidence * 100:.2f}%)")

        # Display the image
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        image_label.config(image=img_tk)
        image_label.image = img_tk

    except Exception as e:
        result_label.config(text=f"Error during processing: {str(e)}")


# Function to browse and select an image file
def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        recognize_image(file_path)


# Create Tkinter window
root = tk.Tk()
root.title("Object Recognition App")

# Create a button to upload an image
browse_button = Button(root, text="Upload Image", command=browse_file)
browse_button.pack()

# Label to display the result
result_label = Label(root, text="Upload an image to start recognition", pady=10)
result_label.pack()

# Label to display the uploaded image
image_label = Label(root)
image_label.pack()

# Run the Tkinter event loop
root.mainloop()
