import cv2
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

def image_matching(master_image, production_image):
    # Convert image data to numpy arrays
    master_img = cv2.imdecode(np.frombuffer(master_image.read(), np.uint8), cv2.IMREAD_COLOR)
    production_img = cv2.imdecode(np.frombuffer(production_image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convert images to grayscale
    master_gray = cv2.cvtColor(master_img, cv2.COLOR_BGR2GRAY)
    production_gray = cv2.cvtColor(production_img, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(master_gray, None)
    kp2, des2 = orb.detectAndCompute(production_gray, None)

    # Create BFMatcher (Brute Force Matcher) object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in ascending order of distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw only good matches, set a threshold for matching accuracy
    good_matches = [match for match in matches if match.distance < 50]

    # Draw matches
    img_matches = cv2.drawMatches(master_gray, kp1, production_gray, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Calculate percentage of match
    total_matches = len(matches)
    good_matches_percentage = (len(good_matches) / total_matches) * 100

    return img_matches, good_matches_percentage

def main():
    st.title("Image Matching App")

    master_image = st.file_uploader("Upload Master Image", type=["jpg", "jpeg", "png"])
    production_image = st.file_uploader("Upload Production Image", type=["jpg", "jpeg", "png"])

    if master_image and production_image:
        if st.button("Run"):
            img_matches, good_matches_percentage = image_matching(master_image, production_image)

            # Display the output using Streamlit
            st.image(img_matches, caption=f'Good Matches Percentage: {good_matches_percentage:.2f}%', use_column_width=True)

if __name__ == "__main__":
    main()
