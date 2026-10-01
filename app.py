import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Cat vs Dog AI",
    page_icon="🐾",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f8f9ff 0%, #eef2ff 100%);
    }

    /* Header */
    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 5px;
        color: #22223b;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 35px;
    }

    /* Upload box */
    .upload-card {
        background: white;
        padding: 28px;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    /* Result card */
    .result-card {
        background: white;
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        margin-top: 20px;
    }

    .prediction {
        font-size: 32px;
        font-weight: 800;
        color: #22223b;
        margin: 10px;
    }

    .confidence {
        font-size: 20px;
        color: #555;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777;
        margin-top: 45px;
        font-size: 14px;
    }

    /* Hide Streamlit menu */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# CNN MODEL
# =========================================================

class SimpleCNN(nn.Module):

    def __init__(self, num_classes=2):

        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                16,
                32,
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
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🐾 Cat vs Dog AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload an image and let the CNN model identify whether it is a Cat or Dog.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

model = load_model()

class_names = ["cat", "dog"]


# =========================================================
# TWO COLUMN LAYOUT
# =========================================================

left_col, right_col = st.columns(
    [1, 1],
    gap="large"
)


# =========================================================
# LEFT SIDE - UPLOAD
# =========================================================

with left_col:

    st.markdown(
        '<div class="upload-card">',
        unsafe_allow_html=True
    )

    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose a Cat or Dog image",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )


# =========================================================
# RIGHT SIDE - PREDICTION
# =========================================================

with right_col:

    st.markdown(
        '<div class="result-card">',
        unsafe_allow_html=True
    )

    st.subheader("🤖 AI Prediction")

    if uploaded_file is None:

        st.info(
            "👆 Upload an image to get the prediction."
        )

    else:

        # Preprocess
        input_tensor = preprocess(image)

        input_batch = input_tensor.unsqueeze(0)

        # Prediction
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

        # Emoji
        if predicted_class == "cat":
            emoji = "🐱"
        else:
            emoji = "🐶"

        st.markdown(
            f'<div class="prediction">'
            f'{emoji} {predicted_class.upper()}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence">'
            f'Confidence: <b>{confidence:.2f}%</b>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.progress(
            int(confidence)
        )

        st.success(
            f"Prediction: {predicted_class.capitalize()}"
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    '🐾 Powered by PyTorch CNN | Cat vs Dog Image Classification'
    '</div>',
    unsafe_allow_html=True
)
