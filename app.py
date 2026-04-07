import streamlit as st
from PIL import Image
import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

st.title("SAM2 Crack Detection")
st.write("Upload an image to detect cracks using SAM2 model.")

# Model configuration
MODEL_CFG = "sam2_hiera_b+"
MODEL_PATH = 'checkpoints/best_model.pt'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    """Load SAM2 model with cached checkpoint."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. Please ensure the model is trained and saved.")
        return None

    try:
        # Build base SAM2 model
        sam2_model = build_sam2(MODEL_CFG, "checkpoints/sam2_hiera_base_plus.pt", device=DEVICE)

        # Load fine-tuned weights
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        sam2_model.load_state_dict(checkpoint['model_state_dict'])
        sam2_model.eval()

        # Create predictor
        predictor = SAM2ImagePredictor(sam2_model)

        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
predictor = load_model()

if predictor is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Load the image
            img = Image.open(uploaded_file)
            img_rgb = np.array(img.convert('RGB'))

            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption='Original Image', use_column_width=True)

            st.write("Processing...")

            # Set image for prediction
            predictor.set_image(img_rgb)

            # Use center point as prompt
            H, W = img_rgb.shape[:2]
            point_coords = np.array([[W // 2, H // 2]])
            point_labels = np.array([1])  # 1 = foreground point

            # Predict mask
            with torch.no_grad():
                masks, scores, logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False,
                )

            # Get the mask and ensure it's boolean
            mask = masks[0].astype(bool)

            # Create visualization
            mask_overlay = img_rgb.copy()
            mask_overlay[mask] = mask_overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5

            # Display result
            with col2:
                st.image(mask_overlay.astype(np.uint8), caption='Crack Detection Result', use_column_width=True)

            # Show statistics
            crack_pixels = np.sum(mask)
            total_pixels = mask.size
            crack_percentage = (crack_pixels / total_pixels) * 100

            st.success("Detection Complete!")
            st.write(f"**Crack Coverage:** {crack_percentage:.2f}% of image")
            st.write(f"**Confidence Score:** {scores[0]:.4f}")

            # Option to download mask
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            st.download_button(
                label="Download Mask",
                data=mask_pil.tobytes(),
                file_name="crack_mask.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Please ensure the uploaded image is valid.")
else:
    st.error("Model could not be loaded. Please check the model path and ensure it exists.")
