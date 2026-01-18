import cv2
import numpy as np
import sys
import os

def visualize_match(img1_path, img2_path, output_path='match_result.png'):
    print(f"🔹 Loading images:\n   1. {img1_path}\n   2. {img2_path}")

    # 1. Load Images
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("❌ Error: One or both image paths are invalid.")
        return

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # 2. Init SIFT
    sift = cv2.SIFT_create()

    # 3. Detect & Compute
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        print("❌ Error: Could not extract features from one of the images.")
        return

    # 4. FLANN Matcher
    index_params = dict(algorithm=1, trees=5) # KD-Tree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # 5. Ratio Test (Lowe's)
    good_matches = []
    matches_mask = [[0, 0] for i in range(len(matches))]

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            matches_mask[i] = [1, 0]
            good_matches.append(m)

    print(f"✅ Found {len(good_matches)} good matches.")

    # 6. Draw Matches
    draw_params = dict(matchColor=(0, 255, 0),       # Green for matches
                       singlePointColor=(255, 0, 0), # Red for non-matched points
                       matchesMask=matches_mask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    result_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    # 7. Save
    cv2.imwrite(output_path, result_img)
    print(f"📸 Visual proof saved to '{output_path}'")

# Usage Example
if __name__ == "__main__":
    # You can change these filenames to test specific images
    visualize_match('fingerprint_1.png', 'fingerprint_rotated.png')