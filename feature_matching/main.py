import cv2
import numpy as np

# Step 1: Load query image (object to find)
# and scene image (where to find it)
query = cv2.imread('../query.jpeg',  cv2.IMREAD_GRAYSCALE)
scene = cv2.imread('../scene.jpeg',  cv2.IMREAD_GRAYSCALE)

scene_color = cv2.imread('../scene.jpeg')

# Step 2: Create ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Step 3: Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(query, None)
kp2, des2 = orb.detectAndCompute(scene, None)

print(f'Keypoints in query: {len(kp1)}')
print(f'Keypoints in scene: {len(kp2)}')

# Step 4: Create matcher and match descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Step 5: Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Step 6: Keep only good matches (top 30%)
good_matches = matches[:int(len(matches) * 0.3)]
print(f'Good matches: {len(good_matches)}')

# Step 7: Draw matches
result = cv2.drawMatches(
    query, kp1,
    scene, kp2,
    good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imshow('Feature Matches', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('matches_output.jpg', result)