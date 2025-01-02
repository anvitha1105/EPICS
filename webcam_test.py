import pyttsx3
import cv2
from ultralytics import YOLO

# Initialize YOLOv8 model (download yolov8n.pt weights if not already done)
model = YOLO('yolov8n.pt')  # Make sure you have this model file

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Adjust volume and speech rate (optional)
engine.setProperty('volume', 1.0)  # Max volume
engine.setProperty('rate', 150)  # Speech speed (lower for slower speech)

# Optional: Choose different voice (male or female)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # 1 for female voice, 0 for male voice

# Function to announce detected objects
def announce_objects(detected_objects):
    if detected_objects:
        objects_summary = ', '.join(detected_objects)
        announcement = f"I see the following objects: {objects_summary}."
        print("Announcing: ", announcement)  # Debug print statement
        engine.say(announcement)  # Queue the text for speech
        engine.runAndWait()  # Speak the text
    else:
        print("No objects detected.")

# Start the webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO model to detect objects
    results = model(frame)
    annotated_frame = results[0].plot()  # Annotate the frame with bounding boxes

    # Extract detected object class IDs and names
    detected_objects = [box.cls for box in results[0].boxes]
    detected_names = [model.names[int(cls)] for cls in detected_objects]

    # Announce detected objects
    announce_objects(detected_names)

    # Display the annotated frame with bounding boxes
    cv2.imshow("YOLOv8 Webcam Object Detection", annotated_frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows





