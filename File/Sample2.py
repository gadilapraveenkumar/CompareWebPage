import argparse

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compare_webpages(img1_path, img2_path, img3_path, threshold):
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Resize to same size
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute Structural Similarity Index (SSIM)
    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image to find regions with differences
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find contours to highlight differences
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fail_count = 0

    for c in contours:
        if cv2.contourArea(c) > 100:  # Ignore small artifacts
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
            fail_count += 1

    # Save visual difference image
    cv2.imwrite(img3_path, img1)

    # Calculate results
    pass_percent = round(score * 100, 2)
    fail_percent = round((1 - score) * 100, 2)
    result = "PASS" if score >= threshold else "FAIL"

    print(f"\nComparison Result: {result}")
    print(f"Matching: {pass_percent}%")
    print(f"Differences: {fail_percent}%")
    print(f"Regions Failed: {fail_count}")
    print(f"Visual diff saved at: {img3_path}")

# Example usage:

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two webpage screenshots and highlight missing or changed parts.")
    parser.add_argument("image1", help="Path to first screenshot (base image)")
    parser.add_argument("image2", help="Path to second screenshot (image to compare)")
    parser.add_argument("output", help="Path to save visual diff image")
    parser.add_argument("threshold", type=float, default=0.95, help="Similarity threshold (0.0 to 1.0)")
    #parser.add_argument("--aligned_output", default=None, help="Optional path to save aligned second image")

    args = parser.parse_args()
    #compare_webpages("C:\\Users\\ADMIN\\Downloads\\Images\\Image1.png","C:\\Users\\ADMIN\\Downloads\\Images\\Image2.png","C:\\Users\\ADMIN\\Downloads\\Images\\visual_diff.png", 0.9)
    compare_webpages(args.image1, args.image2, args.output, args.threshold)
