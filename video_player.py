import cv2

# Create a VideoCapture object to read the video file
cap = cv2.VideoCapture("/home/dslab/Documents/car/cars.mp4")

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Create a window to display the video
cv2.namedWindow("Video Player")

# Start a loop to iterate over the frames of the video
while True:

    # Capture the next frame of the video
    ret, frame = cap.read()

    # If the frame was not captured successfully, break the loop
    if not ret:
        break

    # Resize the frame
    resized_frame = cv2.resize(frame, (640, 480))

    # Display the resized frame in the window
    cv2.imshow("Video Player", resized_frame)

    # Press `q` to quit the program
    if cv2.waitKey(15) & 0xFF == ord('x'):
            break

# Release the VideoCapture object and destroy the window
cap.release()
cv2.destroyAllWindows()
