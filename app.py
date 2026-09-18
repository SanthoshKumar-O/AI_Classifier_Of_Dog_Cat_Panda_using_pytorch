import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# --------------------------------------------------

# Page configuration

# --------------------------------------------------

st.set_page_config(
page_title="Animal Classifier",
page_icon="🐾",
layout="centered"
)

# --------------------------------------------------

# Configuration

# --------------------------------------------------

DEVICE = torch.device(
"cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = ["cats", "dogs", "panda"]

MODEL_PATH = "best_resnet18.pth"

# --------------------------------------------------

# Image preprocessing

# Must match the preprocessing used during training

# --------------------------------------------------

transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)
])

# --------------------------------------------------

# Load trained model

# --------------------------------------------------

@st.cache_resource
def load_model():
    # Recreate the SAME architecture used during training.
    # No pretrained weights are downloaded here.
    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 3)
    )

    # Load the trained weights
    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


model = load_model()

# --------------------------------------------------

# UI

# --------------------------------------------------

st.title("🐾 Animal Image Classifier")

st.write(
"Upload an image and the trained ResNet18 model "
"will classify it as a cat, dog, or panda."
)

st.caption(
"Transfer Learning • ResNet18 • PyTorch"
)

# --------------------------------------------------

# Upload image

# --------------------------------------------------

uploaded_file = st.file_uploader(
"Upload an image",
type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------

# Prediction

# --------------------------------------------------

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=500
    )

    # Convert image to model input
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move to same device as model
    image_tensor = image_tensor.to(DEVICE)

    # ----------------------------------------------
    # Inference
    # ----------------------------------------------

    with torch.no_grad():
        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = probabilities[
            0, predicted_index
        ].item()

    # --------------------------------------------------
    # Display prediction
    # --------------------------------------------------

    st.subheader("Prediction")

    emoji = {
        "cats": "🐱",
        "dogs": "🐶",
        "panda": "🐼"
    }

    st.success(
        f"{emoji[predicted_class]} "
        f"{predicted_class.upper()}"
    )

    st.metric(
        "Confidence",
        f"{confidence * 100:.2f}%"
    )

    # --------------------------------------------------
    # Probability distribution
    # --------------------------------------------------

    st.subheader("Class Probabilities")

    for i, class_name in enumerate(CLASS_NAMES):
        probability = probabilities[
            0, i
        ].item()

        st.write(
            f"**{class_name.capitalize()}** "
            f"{probability * 100:.2f}%"
        )

        st.progress(probability)

# --------------------------------------------------

# Model information

# --------------------------------------------------

with st.expander("About the Model"):
    st.write(
        """
        This application uses a trained ResNet18 model
        with Transfer Learning.

        The original ImageNet classification layer was
        replaced with a custom classifier:

        512 → 256 → ReLU → Dropout → 3

        The convolutional layers were frozen during
        training, and the trained classifier weights
        were saved to best_resnet18.pth.

        The saved weights are loaded when this application
        starts. No training is performed during inference.
        """
    )

    st.write(f"**Device:** {DEVICE}")
    st.write("**Classes:** Cat, Dog, Panda")
    st.write("**Test Accuracy:** 98.67%")

st.divider()

st.caption(
    "Built with PyTorch and Streamlit"
)
