import numpy as np
import tensorflow as tf
import cv2

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details of the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Preprocessing function for a single frame
def preprocess_frame(frame):
    # Convert the image to grayscale if required by the model
    input_shape = input_details[0]['shape']
    if input_shape[-1] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to match the model input size
    height, width = int(input_shape[1]), int(input_shape[2])
    frame_resized = cv2.resize(frame, (width, height))

    # Normalize the frame
    frame_normalized = frame_resized / 255.0

    # Expand dimensions to match the input shape
    if input_shape[-1] == 1:
        input_tensor = np.expand_dims(frame_normalized, axis=-1)  # Add channel dimension for grayscale
    else:
        input_tensor = frame_normalized

    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)  # Add batch dimension
    return input_tensor


# Function to run inference on a frame and return all predictions
def recognize_frame(frame):
    try:
        # Preprocess the frame
        input_tensor = preprocess_frame(frame)

        # Set input tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Load labels
        labels = []
        with open('labels.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        # Pair each label with its confidence score
        predictions = {labels[i]: output_data[0][i] for i in range(len(labels))}

        # Sort predictions by confidence, highest first
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        return sorted_predictions

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return None


# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the live feed
    cv2.imshow('Live Camera - Press 1 to Capture and Predict', frame)

    # Wait for key press and check if '1' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        # Capture the current frame and make prediction
        predictions = recognize_frame(frame)

        if predictions is not None:
            y_offset = 30
            for label, confidence in predictions:
                # Display each label and confidence on the captured frame
                cv2.putText(frame, f"{label}: {confidence * 100:.2f}%", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30

            # Show the captured frame with all predictions
            cv2.imshow('Captured Image - All Predictions', frame)
            cv2.waitKey(0)  # Wait until any key is pressed to close the prediction window

    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
