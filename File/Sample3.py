import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def align_images(img1, img2):
    # Convert to grayscale for feature detection
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        raise ValueError("Not enough matches to align the images.")

    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography and align
    matrix, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    aligned_img2 = cv2.warpPerspective(img2, matrix, (img1.shape[1], img1.shape[0]))

    return aligned_img2

def compare_webpages_for_missing_parts(img1_path, img2_path, diff_img_path="C:\\Users\\ADMIN\\Downloads\\Images\\missing_diff.png", threshold=0.95):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise ValueError("Failed to load one or both images.")

    # Align images
    img2_aligned = align_images(img1, img2)

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    # Threshold diff image to get significant differences
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fail_count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 150:  # Filter small changes
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
            fail_count += 1

    if fail_count > 0:
        cv2.imwrite(diff_img_path, img1)

    result = "PASS" if score >= threshold else "FAIL"
    print(f"\nComparison Result: {result}")
    print(f"Matching Score: {round(score * 100, 2)}%")
    print(f"Missing/Changed Regions: {fail_count}")
    if fail_count > 0:
        print(f"Visual diff saved to: {diff_img_path}")

    return {
        "score": round(score, 4),
        "result": result,
        "missing_regions": fail_count
    }

# Example usage:
compare_webpages_for_missing_parts("C:\\Users\\ADMIN\\Downloads\\Images\\Image1.png", "C:\\Users\\ADMIN\\Downloads\\Images\\Image2.png")
