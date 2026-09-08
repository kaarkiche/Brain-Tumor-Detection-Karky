import os
import time
import numpy as np
import streamlit as st

from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "brain_tumor_efficientnet.keras"
)

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_brain_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    return load_model(MODEL_PATH)


# ============================================================
# CREATE MRI SCAN FRAME
# ============================================================

def create_scan_frame(image, y_position):

    frame = image.copy().convert("RGB")

    draw = ImageDraw.Draw(frame, "RGBA")

    width, height = frame.size

    # Soft scanning region
    top = max(0, y_position - 35)
    bottom = min(height, y_position + 35)

    draw.rectangle(
        (0, top, width, bottom),
        fill=(0, 200, 255, 25)
    )

    # Scanning glow
    for offset, alpha, line_width in [
        (-12, 45, 2),
        (-6, 90, 2),
        (0, 255, 4),
        (6, 90, 2),
        (12, 45, 2),
    ]:

        y = y_position + offset

        if 0 <= y < height:

            draw.line(
                (0, y, width, y),
                fill=(0, 220, 255, alpha),
                width=line_width
            )

    return frame


# ============================================================
# MRI SCANNING ANIMATION
# ============================================================

def run_scan_animation(image):

    st.subheader("🔬 MRI Scan Analysis")

    image_placeholder = st.empty()

    status_placeholder = st.empty()

    progress_placeholder = st.progress(0)

    display_image = image.copy().convert("RGB")

    # Resize large images for smoother animation
    max_width = 700

    if display_image.width > max_width:

        ratio = max_width / display_image.width

        new_size = (
            max_width,
            int(display_image.height * ratio)
        )

        display_image = display_image.resize(new_size)

    height = display_image.height

    total_steps = 55

    for step in range(total_steps):

        y_position = int(
            (step / (total_steps - 1)) * height
        )

        frame = create_scan_frame(
            display_image,
            y_position
        )

        image_placeholder.image(
            frame,
            use_container_width=True
        )

        progress = int(
            ((step + 1) / total_steps) * 100
        )

        progress_placeholder.progress(progress)

        if step < 10:

            status_placeholder.info(
                "🔄 Initializing MRI analysis..."
            )

        elif step < 22:

            status_placeholder.info(
                "🧹 Preprocessing MRI image..."
            )

        elif step < 35:

            status_placeholder.info(
                "🧬 Extracting deep features..."
            )

        elif step < 48:

            status_placeholder.info(
                "🤖 EfficientNetB0 analyzing MRI patterns..."
            )

        else:

            status_placeholder.info(
                "📊 Generating classification result..."
            )

        time.sleep(0.025)

    return image_placeholder


# ============================================================
# PREDICTION
# ============================================================

def predict_image(model, image):

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predictions = predictions[0]

    if len(predictions) != len(CLASS_NAMES):

        raise ValueError(
            "Model output does not contain exactly "
            f"{len(CLASS_NAMES)} classes."
        )

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        predictions
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 NeuroScan AI")

    st.caption(
        "Brain MRI Intelligence Platform"
    )

    st.divider()

    st.subheader("📌 Navigation")

    page = st.radio(
        "Go to",
        [
            "MRI Analysis",
            "Model Information"
        ],
        label_visibility="collapsed"
    )

    # Push About Project toward the bottom
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    st.divider()

    st.caption("📚 Learn more")

    about_project = st.button(
        "ℹ️ About Project",
        use_container_width=True
    )

    if about_project:

        page = "About Project"


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🧠 NeuroScan AI")

st.subheader(
    "Brain Tumor MRI Classification"
)

st.write(
    "An intelligent deep-learning interface for "
    "research-based classification of brain MRI scans."
)

st.divider()


# ============================================================
# MRI ANALYSIS
# ============================================================

if page == "MRI Analysis":

    st.header("📤 Upload MRI Scan")

    st.write(
        "Upload a brain MRI image to begin the AI analysis."
    )

    uploaded_file = st.file_uploader(
        "Choose MRI image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp"
        ],
        help="Supported formats: JPG, JPEG, PNG, BMP and WEBP."
    )

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.success(
                "✅ MRI scan uploaded successfully!"
            )

            st.divider()

            # ====================================================
            # PREVIEW
            # ====================================================

            st.header("🖼️ MRI Preview")

            preview_col1, preview_col2, preview_col3 = st.columns(
                [1, 2, 1]
            )

            with preview_col2:

                st.image(
                    image,
                    caption="Uploaded MRI Scan",
                    use_container_width=True
                )

            st.divider()

            # ====================================================
            # IMAGE DETAILS
            # ====================================================

            st.header("📋 Scan Information")

            info1, info2, info3, info4 = st.columns(4)

            with info1:

                st.metric(
                    "Width",
                    f"{image.width}px"
                )

            with info2:

                st.metric(
                    "Height",
                    f"{image.height}px"
                )

            with info3:

                st.metric(
                    "Color Mode",
                    image.mode
                )

            with info4:

                st.metric(
                    "Model Input",
                    "224 × 224"
                )

            st.divider()

            # ====================================================
            # ANALYZE BUTTON
            # ====================================================

            st.header("🚀 AI Analysis")

            analyze = st.button(
                "🔬 Start MRI Analysis",
                use_container_width=True,
                type="primary"
            )

            if analyze:

                try:

                    # --------------------------------------------
                    # LOAD MODEL
                    # --------------------------------------------

                    with st.spinner(
                        "Loading NeuroScan AI model..."
                    ):

                        model = load_brain_model()

                    # --------------------------------------------
                    # SCAN ANIMATION
                    # --------------------------------------------

                    run_scan_animation(
                        image
                    )

                    # --------------------------------------------
                    # PREDICTION
                    # --------------------------------------------

                    start_time = time.time()

                    (
                        predicted_class,
                        confidence,
                        predictions
                    ) = predict_image(
                        model,
                        image
                    )

                    inference_time = (
                        time.time() - start_time
                    )

                    st.divider()

                    # ====================================================
                    # RESULT
                    # ====================================================

                    st.header(
                        "🧠 AI Classification Result"
                    )

                    result1, result2 = st.columns(2)

                    with result1:

                        st.success(
                            f"### 🎯 Predicted Class\n\n"
                            f"## {predicted_class}"
                        )

                    with result2:

                        st.info(
                            f"### 📊 Confidence\n\n"
                            f"## {confidence * 100:.2f}%"
                        )

                    st.divider()

                    # ====================================================
                    # RESULT METRICS
                    # ====================================================

                    metric1, metric2, metric3 = st.columns(3)

                    with metric1:

                        st.metric(
                            "Classification",
                            predicted_class
                        )

                    with metric2:

                        st.metric(
                            "Confidence",
                            f"{confidence * 100:.2f}%"
                        )

                    with metric3:

                        st.metric(
                            "Inference Time",
                            f"{inference_time:.3f}s"
                        )

                    st.divider()

                    # ====================================================
                    # PROBABILITIES
                    # ====================================================

                    st.header(
                        "📊 Classification Probabilities"
                    )

                    for class_name, probability in zip(
                        CLASS_NAMES,
                        predictions
                    ):

                        percentage = (
                            float(probability) * 100
                        )

                        st.write(
                            f"**{class_name}**"
                        )

                        st.progress(
                            min(
                                int(percentage),
                                100
                            )
                        )

                        st.caption(
                            f"{percentage:.2f}% probability"
                        )

                    st.divider()

                    # ====================================================
                    # SUMMARY
                    # ====================================================

                    st.header(
                        "📌 Analysis Summary"
                    )

                    if predicted_class == "No Tumor":

                        st.success(
                            "The model classified this MRI "
                            "image as **No Tumor**."
                        )

                    elif predicted_class == "Glioma":

                        st.warning(
                            "The model classified this MRI "
                            "image as **Glioma**."
                        )

                    elif predicted_class == "Meningioma":

                        st.warning(
                            "The model classified this MRI "
                            "image as **Meningioma**."
                        )

                    elif predicted_class == "Pituitary":

                        st.warning(
                            "The model classified this MRI "
                            "image as **Pituitary**."
                        )

                    st.info(
                        "ℹ️ This result is generated by an "
                        "AI classification model for research "
                        "and educational purposes. It is not "
                        "a medical diagnosis."
                    )

                except Exception as e:

                    st.error(
                        "❌ Prediction failed."
                    )

                    st.exception(e)

        except Exception as e:

            st.error(
                "❌ Unable to read the uploaded image."
            )

            st.exception(e)

    else:

        st.info(
            "👆 Upload an MRI image above to start."
        )

        st.divider()

        # ====================================================
        # PIPELINE
        # ====================================================

        st.header("⚡ How NeuroScan AI Works")

        step1, step2, step3, step4 = st.columns(4)

        with step1:

            st.info(
                "📤 **Upload**\n\n"
                "Select a brain MRI scan."
            )

        with step2:

            st.info(
                "🧹 **Preprocess**\n\n"
                "Prepare the image for the model."
            )

        with step3:

            st.info(
                "🧠 **Analyze**\n\n"
                "EfficientNetB0 analyzes the scan."
            )

        with step4:

            st.success(
                "📊 **Result**\n\n"
                "View the classification probabilities."
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.header("🤖 Model Information")

    st.write(
        "NeuroScan AI uses EfficientNetB0 with "
        "transfer learning for four-class brain MRI "
        "image classification."
    )

    st.divider()

    info1, info2, info3 = st.columns(3)

    with info1:

        st.metric(
            "Architecture",
            "EfficientNetB0"
        )

    with info2:

        st.metric(
            "Input Resolution",
            "224 × 224"
        )

    with info3:

        st.metric(
            "Output Classes",
            "4"
        )

    st.divider()

    st.header("🧬 Classification Categories")

    class1, class2, class3, class4 = st.columns(4)

    with class1:

        st.warning(
            "### Glioma\n\n"
            "Tumor category"
        )

    with class2:

        st.warning(
            "### Meningioma\n\n"
            "Tumor category"
        )

    with class3:

        st.success(
            "### No Tumor\n\n"
            "Non-tumor category"
        )

    with class4:

        st.warning(
            "### Pituitary\n\n"
            "Tumor category"
        )

    st.divider()

    st.header("⚙️ AI Processing Pipeline")

    st.info("🖼️ MRI Image")

    st.write("↓")

    st.info("📐 Resize to 224 × 224")

    st.write("↓")

    st.info("🧠 EfficientNetB0 Feature Extraction")

    st.write("↓")

    st.info("🔢 Four-Class Classification")

    st.write("↓")

    st.success("📊 Prediction Probability")


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.header("ℹ️ About NeuroScan AI")

    st.write(
        "NeuroScan AI is an academic deep-learning project "
        "designed to demonstrate the application of "
        "computer vision and convolutional neural networks "
        "to brain MRI image classification."
    )

    st.divider()

    st.header("🎯 Project Objective")

    st.write(
        "The objective is to develop an AI-based system "
        "that can classify brain MRI images into four "
        "predefined categories using a trained "
        "EfficientNetB0 model."
    )

    st.divider()

    st.header("🛠️ Technologies Used")

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:

        st.info(
            "🐍 **Python**\n\n"
            "Core programming language"
        )

    with tech2:

        st.info(
            "🧠 **TensorFlow**\n\n"
            "Deep learning framework"
        )

    with tech3:

        st.info(
            "⚡ **Keras**\n\n"
            "Neural network API"
        )

    with tech4:

        st.info(
            "🌐 **Streamlit**\n\n"
            "Web application framework"
        )

    st.divider()

    st.header("📚 Model")

    st.write(
        "The application uses EfficientNetB0 with "
        "transfer learning to extract meaningful "
        "features from MRI images and classify them "
        "into four categories."
    )

    st.divider()

    st.warning(
        "⚠️ Research & Educational Disclaimer"
    )

    st.write(
        "NeuroScan AI is intended for academic, "
        "educational, and research purposes only. "
        "The output of this application should not "
        "be considered a medical diagnosis or used "
        "for clinical decision-making."
    )

    st.divider()

    st.caption(
        "🧠 NeuroScan AI • Brain MRI Deep Learning Project"
    )