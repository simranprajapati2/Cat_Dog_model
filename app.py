
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

# 1. Define the CNN Model (must be identical to the one used for training)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 62 * 62, 128), # Based on 250x250 input after two 2x2 maxpools
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 2. Load the trained model
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model(model_path='simple_cnn_model.pth', num_classes=2):
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # Set to evaluation mode
    return model

# 3. Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(250), # Ensure image is 250x250, though our training data was already this size
    transforms.CenterCrop(250), # Center crop to 250x250
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Streamlit App Interface
st.title('Image Classification: Cat vs. Dog')
st.write('Upload an image to classify if it is a cat or a dog.')

# Load the model
# Assuming you have 2 classes ('cat', 'dog')
# You might need to adjust num_classes if you had more categories
model = load_model(num_classes=2)

# Define class names (must match the order during training)
class_names = ['cat', 'dog'] # Update this if your classes are different

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
  st.image(image, caption="Uploaded Image.", use_container_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model

    # Make prediction
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_probability, predicted_idx = torch.max(probabilities, 0)
        predicted_class = class_names[predicted_idx.item()]

    st.success(f'Prediction: **{predicted_class}** (Confidence: {predicted_probability.item()*100:.2f}%)')
