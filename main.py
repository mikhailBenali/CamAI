import cv2 as cv
from PIL import Image
from transformers import pipeline
import torch

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

device = 0 if torch.cuda.is_available() else -1
detection_model_pipeline = pipeline("object-detection", model="facebook/detr-resnet-50", device=device)
image_captioning_model_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Display the resulting frame
    if cv.waitKey(1) == ord('q'):
        break
    
    # Convert the frame to PIL Image
    pil_frame = Image.fromarray(frame)
    
    # Perform object detection
    result = detection_model_pipeline(pil_frame, batch_size=8)

    # Draw the bounding boxes
    for obj in result:
        x0, y0, x1, y1 = obj["box"].values()
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        cv.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv.putText(frame, obj["label"], (x0, y0), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Perform image captioning
    caption = image_captioning_model_pipeline(pil_frame, batch_size=32)
    print(caption)
    
    cv.imshow('frame', frame)

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
