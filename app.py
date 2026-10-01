import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# =========================================================
# SIMPLE CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    font-size: 17px;
    color: #666666;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #dddddd;
    text-align: center;
    margin-top: 20px;
}

.result-text {
    font-size: 30px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🐾 Cat vs Dog Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Upload an image and let the CNN model predict the animal.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MODEL
# =========================================================

class SimpleCNN(nn.Module):

    def __init__(self, num_classes=2):

        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3, 16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                32 * 62 * 62,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = SimpleCNN(num_classes=2)

    model.load_state_dict(
        torch.load(
            "simple_cnn_model.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model


model = load_model()


# =========================================================
# CLASS NAMES
# =========================================================

class_names = ["cat", "dog"]


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

preprocess = transforms.Compose([

    transforms.Resize((250, 250)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# UPLOAD
# =========================================================

st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.divider()

    # Prediction button
    if st.button(
        "🔍 Predict Image",
        use_container_width=True
    ):

        with st.spinner("Analyzing image..."):

            input_tensor = preprocess(image)

            input_batch = input_tensor.unsqueeze(0)

            with torch.no_grad():

                output = model(input_batch)

                probabilities = torch.nn.functional.softmax(
                    output[0],
                    dim=0
                )

                predicted_probability, predicted_idx = torch.max(
                    probabilities,
                    0
                )

                predicted_class = class_names[
                    predicted_idx.item()
                ]

                confidence = (
                    predicted_probability.item() * 100
                )

        # Result
        if predicted_class == "cat":
            emoji = "🐱"
        else:
            emoji = "🐶"

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-text">'
            f'{emoji} {predicted_class.upper()}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.write(
            f"Confidence: **{confidence:.2f}%**"
        )

        st.progress(
            min(int(confidence), 100)
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

else:

    st.info(
        "👆 Please upload an image to start prediction."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🐾 Cat vs Dog Image Classification • "
    "Built with PyTorch and Streamlit"
)
