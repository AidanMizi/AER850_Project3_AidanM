
# Imports
import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt
import matplotlib.image as mpimg



# Step 1: object masking

# define the path to the input image
motherboard_image_path = r"C:\Users\Aidan Miziolek\Documents\GitHub\AER850_Project3_AidanM\motherboard_image.JPEG"


# load and preprocess the image
img = cv2.imread(motherboard_image_path)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

# edge detection
img_edges = cv2.Canny(blurred_img, 50, 150)

kernel = np.ones((5, 5), np.uint8)
edges_dilated = cv2.dilate(img_edges, kernel, iterations=1)
edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel, iterations=2)


# contour detection
contours, hierarchy = cv2.findContours(
    edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


large_contours = [c for c in contours if cv2.contourArea(c) > 4000]

pcb_contour = max(large_contours, key=cv2.contourArea)

# create mask
mask = np.zeros_like(gray_img)
cv2.drawContours(mask, [pcb_contour], contourIdx=-1, color=255, thickness=-1)


extracted = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

# display everything
plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
plt.title("Original Motherboard Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(gray_img, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(blurred_img, cmap="gray")
plt.title("Blurred Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(img_edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(mask, cmap="gray")
plt.title("Binary Mask (Largest Contour)")
plt.axis("off")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(extracted)
plt.title("Extracted PCB via Mask")
plt.axis("off")
plt.show()


# %%

# Step 2: Yolo V11
model = YOLO("yolo11n.pt")

model.train(data=r"C:\Users\Aidan Miziolek\Documents\GitHub\AER850_Project3_AidanM\Project 3 Data\data\data\data.yaml" , epochs=125, imgsz = 1200, batch = 2, name = 'model_project_3', workers=0, device=0) 





# %%

# Step 3: model evaluation

model.predict(source=r"C:\Users\Aidan Miziolek\Documents\GitHub\AER850_Project3_AidanM\Project 3 Data\data\data\evaluation", conf=0.25, save=True, project=r"C:\Users\Aidan Miziolek\Documents\GitHub\AER850_Project3_AidanM",
             name="evaluation_results")




# conf_matrix = mpimg.imread(r"/content/runs/detect/yolo_model_project_32")
# plt.imshow(conf_matrix)






