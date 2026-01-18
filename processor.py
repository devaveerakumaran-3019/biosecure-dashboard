import cv2
import numpy as np

class ForensicEngine:
    """
    Advanced Fingerprint Engine using SIFT and RANSAC.
    Solves the 'Different Image' problem by verifying geometric structure.
    """

    def __init__(self):
        # SIFT is more accurate than ORB for fingerprints
        # It handles scaling (pressure) and rotation much better.
        self.sift = cv2.SIFT_create()

    def preprocess(self, img):
        """
        Aggressive cleaning to make different images look similar.
        """
        # 1. Resize to standard resolution
        img = cv2.resize(img, (400, 400)) # Slightly smaller helps noise

        # 2. CLAHE (Strong Contrast Enhancement)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        # 3. Gaussian Blur (Removes sensor noise/dirt)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # 4. Adaptive Thresholding (Binarization)
        # This ignores lighting and just focuses on Ridge vs Valley
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return img

    def process_image(self, image_stream):
        """Converts uploaded image stream into SIFT descriptors."""
        data = np.frombuffer(image_stream.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None
            
        # Clean the image first
        processed_img = self.preprocess(img)
        
        # Detect Features
        keypoints, descriptors = self.sift.detectAndCompute(processed_img, None)
        
        if descriptors is None:
            return []
            
        # Serialize Keypoints (x, y) and Descriptors for storage
        # We need Keypoints for RANSAC later!
        serialized_data = []
        for kp, desc in zip(keypoints, descriptors):
            serialized_data.append({
                "pt": kp.pt,         # (x, y) coordinates
                "desc": desc.tolist() # The mathematical vector
            })
            
        return serialized_data

    def match(self, query_data, db_data):
        """
        Advanced Matching using FLANN and RANSAC.
        """
        if not query_data or not db_data:
            return 0, False

        # 1. Reconstruct Data from JSON/Dict format
        # We need numpy arrays for OpenCV
        try:
            kp1 = np.array([q["pt"] for q in query_data])
            des1 = np.array([q["desc"] for q in query_data], dtype=np.float32)
            
            kp2 = np.array([d["pt"] for d in db_data])
            des2 = np.array([d["desc"] for d in db_data], dtype=np.float32)
        except:
            return 0, False

        if len(des1) < 2 or len(des2) < 2:
            return 0, False

        # 2. FLANN Matcher (Fast Library for Approximate Nearest Neighbors)
        # Much faster and accurate for SIFT than BFMatcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # 3. Lowe's Ratio Test
        # Filter out 90% of bad matches immediately
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # 4. GEOMETRIC VERIFICATION (RANSAC)
        # This works even if the image is stretched or rotated differently
        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

            # Find Homography (Transformation matrix)
            # RANSAC will ignore outliers that don't fit the pattern
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if mask is None:
                return 0, False

            matchesMask = mask.ravel().tolist()
            
            # The actual "score" is how many points fit the geometric model
            real_match_count = sum(matchesMask)
            
            # SCORING STRATEGY
            # If > 10 points fit the GEOMETRY perfectly, it is a match.
            score = min((real_match_count / 20) * 100, 100)
            return round(score, 2), real_match_count >= 10

        else:
            return 0, False