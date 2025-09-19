Gradio Link = https://af6d9e42642073e857.gradio.live

🧠 Handwritten Digit Classifier (CNN)

A Convolutional Neural Network (CNN) model built with TensorFlow/Keras to classify handwritten digits (0–9) from the MNIST dataset.
The project also includes a Gradio web interface for real-time predictions via image upload or webcam.

📌 Features

Deep learning model using Conv2D, MaxPooling, Dense, Dropout layers.

Trained on MNIST dataset (28×28 grayscale digit images).

Interactive Gradio UI for predictions.

Supports image preprocessing (grayscale, resizing, normalization).

Achieves high test accuracy (>98%).

🏗 Project Structure
CNN-task/
│── app.py                # Gradio app for digit prediction
│── train.py              # Model training script
│── my_cnn_model.h5       # Saved trained CNN model
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
