import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import io

def compare_images(img1, img2, threshold=0.9):
    # Convert to OpenCV format
    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

    # Resize second image to match the first
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Grayscale conversion
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    # Threshold
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fail_count = 0

    for c in contours:
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
            fail_count += 1

    # Decision
    result = "✅ PASS" if score >= threshold else "❌ FAIL"
    return score, fail_count, result, cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)


# Streamlit UI
st.title("📋 Webpage Screenshot Comparator")
st.markdown("Upload two webpage screenshots to compare layout and elements.")

col1, col2 = st.columns(2)

with col1:
    img1_file = st.file_uploader("Upload Screenshot 1", type=["png", "jpg", "jpeg"])

with col2:
    img2_file = st.file_uploader("Upload Screenshot 2", type=["png", "jpg", "jpeg"])

if img1_file and img2_file:
    img1 = Image.open(img1_file)
    img2 = Image.open(img2_file)

    if st.button("🔍 Compare"):
        with st.spinner("Comparing..."):
            score, fails, result, diff_img = compare_images(img1, img2)

        st.subheader("📊 Result:")
        st.success(result)
        st.write(f"Similarity: **{round(score * 100, 2)}%**")
        st.write(f"Differences Detected: **{fails}**")

        st.image(diff_img, caption="Visual Diff with Differences Highlighted", use_column_width=True)
