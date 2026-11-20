
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms
from model import DigitCNN


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Digit Recognition",
    page_icon="app_icon.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Explicit browser favicon for the tab / browser icon
st.markdown(
    '<link rel="icon" href="app_icon.png" type="image/png">',
    unsafe_allow_html=True
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #0b1422);
    color: #e8eef8;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
h1, h2, h3, p, label {
    color: #e8eef8;
}
.dashboard-card {
    background: linear-gradient(145deg, #101c2c, #0b1422);
    border: 1px solid #30445f;
    border-radius: 15px;
    padding: 24px;
    margin-bottom: 16px;
}
.logo-mark {
    background: transparent !important;
    border-radius: 12px;
    padding: 2px;
}
.logo-mark img {
    background: transparent !important;
    filter: drop-shadow(0 0 8px rgba(22, 219, 145, 0.25));
}
.big-digit {
    color: #16db91;
    font-size: 100px;
    font-weight: 800;
    text-align: center;
    line-height: 1.2;
}
.confidence {
    text-align: center;
    font-size: 19px;
    color: #cbd8eb;
    margin-bottom: 20px;
}
.status {
    color: #16db91;
    font-weight: 600;
}
div.stButton > button {
    border-radius: 10px;
    min-height: 45px;
    border: 1px solid #405570;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


@st.cache_resource
def load_model():
    net = DigitCNN().to(device)
    state = torch.load(
        "mnist_cnn.pth",
        map_location=device,
        weights_only=True
    )
    net.load_state_dict(state)
    net.eval()
    return net



logo_col, title_col = st.columns([0.07, 0.93])

with logo_col:
    st.image("app_icon.png", width=150)

with title_col:
    st.markdown(
        """
        <h2 style="margin-top: 0; margin-bottom: 50px; color: #e8eef8; align-items: center; display: flex;">
            Digit Recognition
        </h2>
        """,
        unsafe_allow_html=True
    )


try:
    model = load_model()
    model_status = True
except Exception as e:
    model = None
    model_status = False
    st.error(
        "Model could not be loaded. Make sure mnist_cnn.pth "
        "is in the project folder and model.py matches your CNN."
    )
    st.exception(e)


# ---------------- CAMERA CONTROLS ----------------
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

camera_col, stop_col, _ = st.columns([1, 1, 3])

with camera_col:
    if st.button("📷 Start Camera", use_container_width=True):
        st.session_state.camera_on = True

with stop_col:
    if st.button("■ Stop Camera", use_container_width=True):
        st.session_state.camera_on = False


# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(pil_image):
    image = ImageOps.grayscale(pil_image)
    image = ImageOps.autocontrast(image, cutoff=1)
    array = np.asarray(image, dtype=np.float32)

    height, width = array.shape
    border_size = max(2, int(min(height, width) * 0.08))
    border = np.concatenate((
        array[:border_size, :].ravel(),
        array[-border_size:, :].ravel(),
        array[:, :border_size].ravel(),
        array[:, -border_size:].ravel(),
    ))

    # Convert both white-paper digits and MNIST-style digits to
    # bright-foreground/dark-background format.
    if float(border.mean()) > 127:
        image = ImageOps.invert(image)
        array = np.asarray(image, dtype=np.float32)

    # Remove accidental border/shadow noise, then crop the digit tightly.
    array = np.clip((array - 20) / (235 - 20), 0, 1)
    foreground = array > 0.18
    if foreground.any():
        ys, xs = np.where(foreground)
        margin = 5
        left = max(0, int(xs.min()) - margin)
        top = max(0, int(ys.min()) - margin)
        right = min(width, int(xs.max()) + margin + 1)
        bottom = min(height, int(ys.max()) + margin + 1)
        array = array[top:bottom, left:right]

    crop_height, crop_width = array.shape
    scale = 20.0 / max(crop_height, crop_width)
    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))

    digit = Image.fromarray(
        np.uint8(array * 255),
        mode="L",
    ).resize(
        (resized_width, resized_height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("L", (28, 28), color=0)
    canvas.paste(digit, ((28 - resized_width) // 2, (28 - resized_height) // 2))

    tensor = transforms.ToTensor()(canvas).unsqueeze(0).to(device)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor, canvas


# ---------------- DASHBOARD LAYOUT ----------------
left, right = st.columns([1.9, 1], gap="large")

with left:
    st.markdown(
        "<div class='dashboard-card'>"
        "<h3>📥 Image Input</h3>"
        "<p style='color:#9fb2cc'>"
        "Use the camera or upload a clear handwritten-digit image."
        "</p></div>",
        unsafe_allow_html=True
    )

    camera_tab, upload_tab = st.tabs(["Camera", "Upload Image"])
    captured = None

    with camera_tab:
        if st.session_state.camera_on:
            captured = st.camera_input(
                "Capture a handwritten digit",
                key="digit_camera"
            )
        else:
            st.info(
                "Press Start Camera, allow browser camera access, "
                "and capture a digit on paper."
            )

    with upload_tab:
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="digit_upload"
        )
        if uploaded is not None:
            captured = uploaded

    if captured is not None:
        pil_image = Image.open(captured).convert("RGB")
        st.image(
            pil_image,
            caption="Source image",
            use_container_width=True
        )

    st.caption(
        "Tip: use a thick dark marker on plain white paper, "
        "with one large digit centered in the frame."
    )


# ---------------- PREDICTION ----------------
prediction = None
confidence = None
probabilities = None
processed_image = None

if model is not None and captured is not None:
    pil_image = Image.open(captured).convert("RGB")
    input_tensor, processed_image = preprocess_image(pil_image)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)[0]
        confidence, prediction = torch.max(probabilities, dim=0)

    prediction = int(prediction.item())
    confidence = float(confidence.item()) * 100


with right:
    st.markdown(
        "<div class='dashboard-card'>"
        "<h3>Prediction</h3>",
        unsafe_allow_html=True
    )

    if prediction is not None:
        st.markdown(
            f"<div class='big-digit'>{prediction}</div>"
            f"<div class='confidence'>Confidence: "
            f"<b>{confidence:.1f}%</b></div>",
            unsafe_allow_html=True
        )

        st.markdown("#### Top 3 Predictions")

        top_values, top_indices = torch.topk(
            probabilities, k=3
        )

        for rank, (prob, digit) in enumerate(
            zip(top_values.tolist(), top_indices.tolist()), start=1
        ):
            percentage = prob * 100

            st.markdown(
                f"**{rank}. &nbsp; {digit}**"
                f"&nbsp;&nbsp; {percentage:.1f}%"
            )
            st.progress(float(prob))

        with st.expander("View processed 28 × 28 image"):
            st.image(
                processed_image,
                caption="Preprocessed model input",
                width=180
            )

    else:
        st.markdown(
            "<div class='big-digit'>—</div>"
            "<div class='confidence'>Waiting for an image</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("#### How it works")
    st.markdown(
        "1. Capture an image from the camera\n\n"
        "2. Convert it to grayscale and resize to 28 × 28\n\n"
        "3. Predict using your trained CNN\n\n"
        "4. Display the predicted digit and probabilities"
    )

    if model_status:
        st.markdown(
            "<p class='status'>● Model loaded successfully</p>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
