import argparse
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def compare_webpages(img1_path, img2_path, img3_path, threshold):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fail_count = 0
    for c in contours:
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
            fail_count += 1

    cv2.imwrite(img3_path, img1)

    pass_percent = round(score * 100, 2)
    fail_percent = round((1 - score) * 100, 2)
    result = "PASS" if score >= threshold else "FAIL"

    # 📝 Generate note file path based on image3_path
    note_path = os.path.splitext(img3_path)[0] + "_note.txt"

    with open(note_path, "w") as note:
        note.write(f"Comparison Result: {result}\n")
        note.write(f"Matching: {pass_percent}%\n")
        note.write(f"Differences: {fail_percent}%\n")
        note.write(f"Regions Failed: {fail_count}\n")
        note.write(f"Visual diff saved at: {img3_path}\n")

    print(f"Note saved to: {note_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two webpage screenshots and highlight missing or changed parts.")
    parser.add_argument("image1", help="Path to first screenshot (base image)")
    parser.add_argument("image2", help="Path to second screenshot (image to compare)")
    parser.add_argument("output", help="Path to save visual diff image")
    parser.add_argument("threshold", type=float, help="Similarity threshold (0.0 to 1.0)")

    args = parser.parse_args()
    compare_webpages(args.image1, args.image2, args.output, args.threshold)
