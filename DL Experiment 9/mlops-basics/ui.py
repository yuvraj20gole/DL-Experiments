import os
import streamlit as st
import requests

st.set_page_config(page_title="Digit Recognition", page_icon="🔢")

API_URL = os.environ.get("API_URL", "http://127.0.0.1:5002/predict")

st.title("Digit Recognition - Deep Learning Lab")
st.write("Upload a handwritten digit image (MNIST-style) to get predictions from the trained baseline ANN model.")

uploaded_file = st.file_uploader(
    "Choose a digit image...",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=150)

    if st.button("Predict"):
        with st.spinner("Classifying digit..."):
            try:
                # Seek to start in case it was read previously
                uploaded_file.seek(0)
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                response = requests.post(API_URL, files=files, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    predicted_digit = data.get("predicted_digit")
                    confidence = data.get("confidence", 0.0)
                    confidence_pct = confidence * 100.0

                    st.success(f"### Predicted Digit: **{predicted_digit}**")
                    st.info(f"Confidence: **{confidence_pct:.2f}%** ({confidence})")
                else:
                    error_msg = response.json().get("error", response.text)
                    st.error(f"API Error ({response.status_code}): {error_msg}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to Flask API. Ensure `python app.py` is running on http://127.0.0.1:5002.")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
