import cv2 as cv
from PIL import Image
from transformers import pipeline
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import numpy as np

pipe = pipeline("object-detection", model="facebook/detr-resnet-50")

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
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

    #Convert the frame to PIL Image
    frame = Image.fromarray(frame)
    
    # you can specify the revision tag if you don't want the timm dependency
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to("cuda")

    inputs = processor(images=frame, return_tensors="pt").to("cuda")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([frame.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        """box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )"""
        # Display a rectangle around the detected object
        frame = np.array(frame)
        cv.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv.putText(frame, model.config.id2label[label.item()], (int(box[0]), int(box[1])), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    
    
    cv.imshow('frame', frame)



# When everything done, release the capture
cap.release()
cv.destroyAllWindows()