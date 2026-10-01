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
    layout="centered"
)


# =========================================================
# SIMPLE UI
# =========================================================

st.title("🐾 Cat vs Dog AI Classifier")

st.write(
    "Upload a cat or dog image and let the trained CNN model predict it."
)

st.divider()


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

    state_dict = torch.load(
        "simple_cnn_model.pth",
        map_location=torch.device("cpu")
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

try:

    model = load_model()

except Exception as e:

    st.error("❌ Model could not be loaded.")

    st.code(str(e))

    st.stop()


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "cat",
    "dog"
]


# =========================================================
# PREPROCESSING
# =========================================================

# IMPORTANT:
# This preprocessing should match the training preprocessing.

preprocess = transforms.Compose([

    transforms.Resize((250, 250)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# =========================================================
# IMAGE UPLOAD
# =========================================================

st.subheader("📷 Upload Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.divider()

    # Predict button
    if st.button(
        "🔍 Predict",
        use_container_width=True
    ):

        with st.spinner("Analyzing image..."):

            # -----------------------------------------
            # PREPROCESS IMAGE
            # -----------------------------------------

            input_tensor = preprocess(image)

            # Add batch dimension
            input_batch = input_tensor.unsqueeze(0)

            # -----------------------------------------
            # MODEL PREDICTION
            # -----------------------------------------

            with torch.no_grad():

                output = model(input_batch)

                probabilities = torch.softmax(
                    output,
                    dim=1
                )

            # -----------------------------------------
            # GET PREDICTION
            # -----------------------------------------

            predicted_index = torch.argmax(
                probabilities,
                dim=1
            ).item()

            predicted_class = class_names[
                predicted_index
            ]

            confidence = probabilities[
                0,
                predicted_index
            ].item() * 100

            # -----------------------------------------
            # GET BOTH PROBABILITIES
            # -----------------------------------------

            cat_probability = (
                probabilities[0][0].item() * 100
            )

            dog_probability = (
                probabilities[0][1].item() * 100
            )

        # =================================================
        # RESULT
        # =================================================

        st.subheader("🤖 Prediction Result")

        if predicted_class == "cat":

            st.success(
                f"🐱 CAT — {confidence:.2f}% confidence"
            )

        else:

            st.success(
                f"🐶 DOG — {confidence:.2f}% confidence"
            )

        # =================================================
        # PROBABILITIES
        # =================================================

        st.write("### 📊 Prediction Probabilities")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🐱 Cat",
                f"{cat_probability:.2f}%"
            )

            st.progress(
                int(cat_probability)
            )

        with col2:

            st.metric(
                "🐶 Dog",
                f"{dog_probability:.2f}%"
            )

            st.progress(
                int(dog_probability)
            )

        st.divider()

        st.caption(
            "Model: SimpleCNN • PyTorch • Input Size: 250 × 250"
        )

else:

    st.info(
        "👆 Upload an image to start prediction."
    )
