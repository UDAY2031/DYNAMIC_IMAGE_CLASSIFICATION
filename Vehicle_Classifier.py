import cv2
from ultralytics import YOLO

# Load your YOLOv8 model
model_path = 'best.pt'
model = YOLO(model_path)

# Path to your video file
video_path = '6.mp4'

# Open a connection to the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    # Read a frame from the video file
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to capture image.")
        break
    
    # Perform inference
    results = model(frame)
    
    # Check results type
    if isinstance(results, list):
        # Access the first result
        result = results[0]
        # Annotate the frame with predictions
        annotated_frame = result.plot()  # Use plot method to get the annotated frame
    else:
        # Handle unexpected result type
        print("Unexpected results type.")
        continue
    
    # Display the frame with annotations
    cv2.imshow('YOLOv8 Video Detection', annotated_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

