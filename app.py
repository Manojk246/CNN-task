import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps

# Load trained CNN model
model = load_model("my_cnn_model.h5")

# Define classes (0-9 digits)
classes = [str(i) for i in range(10)]

# Prediction function
def predict_digit(img: Image.Image):
    # Convert to grayscale
    img = img.convert("L")
    
    # Invert colors (white background, black digit)
    img = ImageOps.invert(img)
    
    # Resize to 28x28 (MNIST format)
    img = img.resize((28, 28))
    
    # Convert to array and normalize
    img_array = img_to_array(img) / 255.0
    
    # Expand dimensions → (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    return {classes[i]: float(pred[0][i]) for i in range(10)}

# Gradio interface
iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil", image_mode="L", sources=["upload", "webcam"]),
    outputs=gr.Label(num_top_classes=3),
    title="🧠 Handwritten Digit Classifier (0-9)",
    description="Upload or take a photo of a digit, and the CNN will classify it."
)

# Run the app
if __name__ == "__main__":
    iface.launch()
