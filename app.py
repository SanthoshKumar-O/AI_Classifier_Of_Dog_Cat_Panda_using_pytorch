import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Class names
# -----------------------------
class_names = ["cats", "dogs", "panda"]


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 3)
    )

    model.load_state_dict(
        torch.load(
            "best_resnet18.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model


model = load_model()


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🐱🐶🐼 Animal Image Classifier")

st.write(
    "Upload an image and the trained ResNet18 model "
    "will classify it as a cat, dog, or panda."
)


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=400
    )


    # -------------------------
    # Preprocess
    # -------------------------
    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)


    # -------------------------
    # Prediction
    # -------------------------
    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0, predicted_class
        ].item()


    # -------------------------
    # Display result
    # -------------------------
    predicted_name = class_names[predicted_class]

    st.success(
        f"Prediction: {predicted_name.upper()}"
    )

    st.write(
        f"Confidence: {confidence * 100:.2f}%"
    )


    # -------------------------
    # Show all probabilities
    # -------------------------
    st.subheader("Class Probabilities")

    for i, class_name in enumerate(class_names):

        probability = probabilities[0, i].item()

        st.write(
            f"{class_name}: "
            f"{probability * 100:.2f}%"
        )

        st.progress(probability)