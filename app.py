import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import ImageClassifier

st.set_page_config(page_title="PetVision")

st.title("🐱🐶 PetVision")
st.write("Upload an image and the AI will predict whether it is a Cat or a Dog.")


device = "cuda" if torch.cuda.is_available() else "cpu"



model = ImageClassifier()

model.load_state_dict(
    torch.load(
        "saved_models/image_classifier.pth",
        map_location=device
    )
)

model.to(device)
model.eval()



transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])



class_names = ["Cat","Dog"]



uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():

        output = model(image_tensor)

        probabilities = torch.softmax(output, dim=1)

        confidence, prediction = torch.max(probabilities, dim=1)

    st.success(
        f"Prediction : {class_names[prediction.item()]}"
    )

    st.info(
        f"Confidence : {confidence.item()*100:.2f}%"
    )