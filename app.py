import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import timm
import os
import matplotlib.pyplot as plt
from torchvision import models


# --------- Page Configuration ---------
st.set_page_config(
    page_title="COVID‑19 / Pneumonia Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------- Device ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Sidebar Controls ---------
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    st.write("### Choose model backbone:")
    
    # Create three columns: Left label, toggle, Right label
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Toggle in the middle
    with col2:
        use_efficientnet = st.toggle(" ", label_visibility="collapsed")
        
    # Display labels and highlight the selected model
    with col1:
        st.markdown(
            f"<div style='text-align: center; font-weight: {'bold' if not use_efficientnet else 'normal'};'>ResNet50</div>",
            unsafe_allow_html=True,
        )
        
    with col3:
        st.markdown(
            f"<div style='text-align: center; font-weight: {'bold' if use_efficientnet else 'normal'};'>EfficientNet-B3</div>",
            unsafe_allow_html=True,
        )
        
    # Map to actual model name
    model_name = "EfficientNet-B3" if use_efficientnet else "ResNet50"
    
    # Optional: show selected model
    st.write(f"**Selected model:** {model_name}")
    
    st.markdown("---")
    st.markdown("**EMERGENCY CONTACTS**  \n"
                "• Call: 911  \n"
                "• CDC Hotline: 1‑800‑232‑4636  \n"
                "• Medicare Helpline: 1‑800‑633‑4227")
    st.markdown("---")
    st.markdown("APP BUILT BY TEAM 1")

# --------- Classifier Head ---------
class COVIDNetClassifierLarge(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(COVIDNetClassifierLarge, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# --------- Preprocessing ---------
def get_transform(model_name):
    size = 300 if model_name == 'EfficientNet-B3' else 224
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

# --------- Prediction Function ---------
def predict(image, model, backbone, transform, class_names):
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        backbone.eval()
        features = backbone(input_tensor)
        features_flat = features.view(1, -1)

        model.eval()
        outputs = model(features_flat.to(device))
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return pred, probs[0][pred].item()

# --------- Load Backbone and Classifier Based on Model Selection ---------
@st.cache_resource
def load_model(model_name):
    if model_name == 'EfficientNet-B3':
        backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0).to(device)
        input_dim = 1536
        model_path = "efficientnet_covidnet_large_3.pt"
    else:
        backbone = models.resnet50(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove fc layer
        backbone = nn.Sequential(backbone, nn.Flatten()).to(device)
        input_dim = 2048
        model_path = "resnet_covidnet_large_2.pt"

    model = COVIDNetClassifierLarge(input_dim=input_dim, num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return backbone, model, get_transform(model_name)

# --------- Streamlit App ---------

# Load selected model
backbone, model, transform = load_model(model_name)
class_names = ['COVID-19', 'Pneumonia', 'Normal']

# --------- Tabs ---------
tab_infer, tab_about = st.tabs(["🔍 Inference", "ℹ️ About"])

# --------- Inference Tab ---------
with tab_infer:
    st.header("Chest X‑Ray Diagnosis")
    st.write("Upload a chest X‑ray and check whether it’s COVID‑19, Pneumonia, or Normal.")

    uploaded_file = st.file_uploader("📤 Upload an X‑ray image", type=["png", "jpg", "jpeg"])
    mapping_file = st.file_uploader("🗂️ (Optional) Upload test_mapping.csv", type=["csv"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖼️ Input X‑ray")
            st.image(image)

        with col2:
            st.subheader("⚙️ Running inference…")
            with st.spinner("Classifying…"):
                pred, probs = predict(image, model, backbone, transform, class_names)

            st.subheader("🔍 Prediction Result:")
            st.write(f"✅ **Predicted:** {class_names[pred]} with confidence = ({probs * 100:.2f}%)")

            if class_names[pred] == 'COVID-19':
                st.error("🚨 COVID‑19 detected!")
                st.markdown("**Immediate Actions & Helplines:**")
                st.markdown(
                    "- 📞 Call: 911 (Emergency)\n"
                    "- 📞 CDC Hotline: 1‑800‑232‑4636\n"
                    "- 📞 Medicare Helpline: 1‑800‑633‑4227\n"
                    "- 🏥 Contact your local health department"
                )
            elif class_names[pred] == 'Pneumonia':
                st.warning("⚠️ Pneumonia detected.")
                st.markdown("**Recommended Next Steps:**")
                st.markdown(
                    "- 📞 Contact your primary care provider\n"
                    "- 🏥 Visit urgent care or hospital\n"
                    "- 💊 Follow medical instructions"
                )
            else:
                st.success("✅ No abnormalities detected.")
                st.markdown("**Health Tips:**")
                st.markdown(
                    "- 😊 Stay active and maintain good posture\n"
                    "- 😷 Use masks in crowded places\n"
                    "- 💉 Stay updated on vaccinations"
                )

        if mapping_file is not None:
            try:
                df = pd.read_csv(mapping_file, header=None, names=["image", "label"])
                row = df[df["image"] == uploaded_file.name]
                if not row.empty:
                    actual = class_names[int(row.iloc[0]["label"])]
                    st.info(f"🎯 Actual Label: {actual}")
                else:
                    st.warning("Image not found in mapping CSV.")
            except Exception as e:
                st.error(f"Error reading mapping file: {e}")
                
    st.markdown("---")
    st.markdown(
    "<div style='text-align: center; font-size: 0.9em; color: gray;'>"
    "⚠️ <strong>Disclaimer:</strong> This tool is for educational and research purposes only. "
    "It is <strong>not</strong> intended to provide a professional medical diagnosis or substitute clinical judgment. "
    "Always consult a licensed healthcare provider for any medical concerns."
    "</div>",
    unsafe_allow_html=True)

# --------- About Tab ---------
with tab_about:
    st.markdown("## About this prototype")
    st.write("""
    This web app enables automated diagnosis of chest X-rays using deep learning. The system flags potential cases of COVID-19, Pneumonia, or Normal conditions.
    
    **Pipeline Overview**
    
    1. Image Preprocessing: Resize to model-specific input size (224 or 300 pixels) based on backbone model and normalize pixel values using standard ImageNet statistics
    2. Extract deep features from the X-ray image using the pretrained CNN (ResNet50 or EfficientNet-B3)  
    3. Classify using custom-trained COVID-Net model. 

    """)
