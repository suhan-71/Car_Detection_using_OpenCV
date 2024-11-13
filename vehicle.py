import cv2
import numpy as np

# Video input (0 for webcam, or provide a video file path)
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# Background subtractor object
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Define area of interest and detection line
detect_line_y = 300  # Y-coordinate for detection line
offset = 6  # Allowable error in vehicle detection

# Create kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Initialize vehicle counter
vehicle_count = 0

# Function to detect and count vehicles
def detect_vehicles(frame):
    global vehicle_count
    
    # Apply background subtraction
    fg_mask = background_subtractor.apply(frame)
    
    # Remove shadows and noise using morphological operations
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours (boundaries of the objects detected)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter out small objects
        if cv2.contourArea(contour) > 500:
            # Get bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Get the center point of the bounding box
            center_y = y + h // 2
            
            # Detect vehicles crossing the detection line
            if detect_line_y - offset < center_y < detect_line_y + offset:
                vehicle_count += 1
                cv2.line(frame, (0, detect_line_y), (frame.shape[1], detect_line_y), (0, 0, 255), 3)
                
    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame for consistent processing
    frame = cv2.resize(frame, (640, 480))
    
    # Detect vehicles
    frame = detect_vehicles(frame)
    
    # Draw detection line
    cv2.line(frame, (0, detect_line_y), (frame.shape[1], detect_line_y), (255, 0, 0), 2)
    
    # Display vehicle count on the frame
    cv2.putText(frame, f'Vehicle Count: {vehicle_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display result
    cv2.imshow('Vehicle Detection and Counting', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()