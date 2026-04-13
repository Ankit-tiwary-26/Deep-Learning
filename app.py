import streamlit as st
from PIL import Image
from model import predict_digit

st.set_page_config(page_title="Digit Classifier", layout="centered")

st.title("✍️ Handwritten Digit Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("📷 Uploaded Image")
    st.image(image, width=150)

    digit, confidence, processed = predict_digit(image)

    st.subheader("🧠 Processed Image (Model Input)")
    st.image(processed.reshape(28, 28), width=150, clamp=True)

    st.subheader("✅ Prediction")
    st.success(f"Digit: {digit}")
    st.info(f"Confidence: {confidence:.2f}")