
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Lung Cancer Detection System",
    page_icon="🫁",
    layout="wide"
)

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
IMG_SIZE = (224, 224)
MODEL_PATH = "final_lung_cancer_mobilenetv2_model.keras"

class_names = [
    "adenocarcinoma",
    "large_cell_carcinoma",
    "normal",
    "squamous_cell_carcinoma"
]

cancer_classes = [
    "adenocarcinoma",
    "large_cell_carcinoma",
    "squamous_cell_carcinoma"
]

class_display_names = {
    "adenocarcinoma": "Adenocarcinoma",
    "large_cell_carcinoma": "Large Cell Carcinoma",
    "normal": "Normal / No Cancer Detected",
    "squamous_cell_carcinoma": "Squamous Cell Carcinoma"
}

# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------
@st.cache_resource
def load_lung_model():
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        model = keras.models.load_model(MODEL_PATH)
    except Exception:
        model = keras.models.load_model(MODEL_PATH, safe_mode=False)

    return model


model = load_lung_model()

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image)
    image_array = image_array.astype("float32") / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_image(image):
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)[0]

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    normal_probability = prediction[class_names.index("normal")] * 100

    cancer_probability = 0
    for i, prob in enumerate(prediction):
        if class_names[i] in cancer_classes:
            cancer_probability += prob

    cancer_probability = cancer_probability * 100

    if predicted_class == "normal":
        final_result = "No Lung Cancer Detected by Model"
        result_type = "normal"
    else:
        final_result = "Lung Cancer Suspected by Model"
        result_type = "cancer"

    if confidence >= 80:
        confidence_level = "High Confidence"
    elif confidence >= 60:
        confidence_level = "Medium Confidence"
    else:
        confidence_level = "Low Confidence / Uncertain Prediction"

    probability_df = pd.DataFrame({
        "Class": [class_display_names[name] for name in class_names],
        "Probability (%)": [round(prob * 100, 2) for prob in prediction]
    })

    return {
        "final_result": final_result,
        "result_type": result_type,
        "predicted_class": predicted_class,
        "predicted_class_display": class_display_names[predicted_class],
        "confidence": confidence,
        "confidence_level": confidence_level,
        "normal_probability": normal_probability,
        "cancer_probability": cancer_probability,
        "probability_df": probability_df
    }


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("🫁 Lung Cancer Detection")
st.sidebar.markdown("---")

st.sidebar.info(
    """
    This dashboard uses a MobileNetV2 Transfer Learning CNN model
    to classify lung CT scan images.
    """
)

st.sidebar.markdown("### Prediction Classes")
st.sidebar.write("1. Normal")
st.sidebar.write("2. Adenocarcinoma")
st.sidebar.write("3. Large Cell Carcinoma")
st.sidebar.write("4. Squamous Cell Carcinoma")

st.sidebar.markdown("---")
st.sidebar.warning(
    """
    Academic prototype only.
    Not for real medical diagnosis.
    """
)

# ---------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------
st.title("🫁 Lung Cancer Detection Using CT Scan Images")

st.markdown(
    """
    Upload a lung CT scan image. The system will predict whether the image appears
    **normal** or **lung cancer suspected**.
    If cancer is suspected, it will also show the possible cancer subtype.
    """
)

st.markdown("---")

# ---------------------------------------------------------
# Model Check
# ---------------------------------------------------------
if model is None:
    st.error(
        f"""
        Model file not found.

        Please make sure this file exists in the same Colab directory:

        `{MODEL_PATH}`
        """
    )
    st.stop()

# ---------------------------------------------------------
# Upload Image
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CT Scan Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Uploaded CT Scan Image")
        st.image(image, caption="Input CT Scan", use_container_width=True)

    with right_col:
        st.subheader("Prediction Result")

        result = predict_image(image)

        if result["result_type"] == "cancer":
            st.error(result["final_result"])
        else:
            st.success(result["final_result"])

        st.metric(
            label="Prediction Confidence",
            value=f"{result['confidence']:.2f}%"
        )

        st.write("### Detailed Result")
        st.write(f"**Predicted Class:** {result['predicted_class_display']}")
        st.write(f"**Confidence Level:** {result['confidence_level']}")

        st.write("### Cancer vs Normal Probability")
        st.write(f"**Cancer Probability:** {result['cancer_probability']:.2f}%")
        st.write(f"**Normal Probability:** {result['normal_probability']:.2f}%")

    st.markdown("---")

    st.subheader("Class-wise Prediction Probabilities")
    st.dataframe(result["probability_df"], use_container_width=True)

    st.subheader("Probability Chart")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        result["probability_df"]["Class"],
        result["probability_df"]["Probability (%)"]
    )
    ax.set_ylabel("Probability (%)")
    ax.set_xlabel("Class")
    ax.set_title("Class-wise Prediction Probability")
    plt.xticks(rotation=25, ha="right")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("How to Understand the Result")

    if result["result_type"] == "cancer":
        st.write(
            """
            The model predicted one of the cancer-related classes.
            Therefore, the final result is shown as **Lung Cancer Suspected by Model**.
            """
        )
    else:
        st.write(
            """
            The model predicted the image as **Normal**.
            Therefore, the final result is shown as **No Lung Cancer Detected by Model**.
            """
        )

    st.info(
        """
        Adenocarcinoma, Large Cell Carcinoma, and Squamous Cell Carcinoma are cancer-related classes.
        Normal means the model did not detect lung cancer patterns in the uploaded CT image.
        """
    )

    st.warning(
        """
        Important: This is an academic AI prototype only.
        It must not be used for real medical diagnosis.
        A qualified doctor or radiologist must confirm any medical decision.
        """
    )

else:
    st.info("Please upload a CT scan image to get prediction.")

    st.markdown(
        """
        ### Expected Input
        Upload a CT scan image in one of these formats:

        - JPG
        - JPEG
        - PNG

        ### Expected Output
        The dashboard will show:

        - Lung cancer suspected or no lung cancer detected
        - Detailed predicted class
        - Confidence score
        - Cancer probability
        - Normal probability
        - Class-wise probability chart
        """
    )
