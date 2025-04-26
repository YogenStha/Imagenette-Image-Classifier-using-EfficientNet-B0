import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
import timm

# ----- SETTINGS -----
MODEL_PATH = "best_model.pth"  # Change this to your model's path
NUM_CLASSES = 10  # ImageNette has 10 classes

# ----- LOAD MODEL -----
def load_model():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# ----- IMAGE PREPROCESSING -----
transform = T.Compose([
    T.Resize((256,256)),
    T.CenterCrop((224,224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
])

# ----- CLASS LABELS -----
imagenette_classes = [
    "tench", "English springer", "cassette player", "chain saw",
    "church", "French horn", "garbage truck", "gas pump",
    "golf ball", "parachute"
]

st.title("🧠 ImageNette Classifier")
st.markdown("Upload an image and let the AI predict what it sees!")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing the image..."):
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = output.max(1)
            predicted_class = imagenette_classes[predicted.item()]

    st.success(f"🎯 Prediction: **{predicted_class}**")

    st.markdown("---")
    st.markdown("Made with ❤️ using [Streamlit](https://streamlit.io/) & [PyTorch](https://pytorch.org/)")
