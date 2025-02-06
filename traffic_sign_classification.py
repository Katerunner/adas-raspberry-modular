import cv2
import glob
import random
from ultralytics import YOLO

# Load the classification model
model = YOLO("trained_models/traffic-sign-class-yolov11n.pt")

# Get the list of all image paths in the folder
image_paths = glob.glob("assets/images/traffic_signs/*.jpg")

if not image_paths:
    raise ValueError("No images found in assets/images/traffic_signs/")

while True:
    # Pick a random image
    img_path = random.choice(image_paths)

    # Read the image (OpenCV loads in BGR format)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue  # Skip if image reading failed

    # Convert image to RGB as the model expects RGB input
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Run prediction using the model
    results = model.predict(img_rgb, verbose=False)
    if not results:
        predicted_class = "Unknown"
    else:
        # Use the top1 attribute to get the highest probability class index.
        pred_idx = results[0].probs.top1  # top1 is an integer index of the highest probability
        # Map the index to a class name if available.
        predicted_class = model.names[pred_idx] if hasattr(model, "names") else str(pred_idx)

    # Resize the image to 640x640 pixels
    img_bgr = cv2.resize(img_bgr, (640, 640))

    # Overlay the predicted class text on the image
    text = f"Predicted: {predicted_class}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_color = (0, 255, 0)  # Green text
    cv2.putText(img_bgr, text, (10, 30), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Display the image in an OpenCV window
    cv2.imshow("Traffic Sign Classification", img_bgr)

    # Wait for 2000 ms (2 seconds) or until 'q' is pressed
    key = cv2.waitKey(2000)
    if key & 0xFF == ord('q'):
        break

# Clean up windows
cv2.destroyAllWindows()
