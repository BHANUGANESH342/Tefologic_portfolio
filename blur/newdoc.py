import cv2

# Load the cascade files for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier(r"C:\Users\rkssp\Desktop\virtual envi\face_blur\blur\face blur\haarcascade_mcs_eyepair_big.xml")

# Function to select the detector type
def select_detector():
    while True:
        choice = input("Select detector type (1 for face, 2 for eyes): ")
        if choice == '1' or choice == '2':
            return int(choice)
        else:
            print("Invalid choice. Please enter 1 or 2.")

# Select the detector type
detector_type = select_detector()

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, img = cap.read()

    # Choose the cascade classifier based on user's choice
    if detector_type == 1:
        cascade = face_cascade
    else:
        cascade = eyes_cascade

    # Detect faces or eyes
    detections = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=4)

    if detector_type == 1:
        # Face detection: add a box around the face
        for (x, y, w, h) in detections:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            roi = img[y:y + h, x:x + w]
            blur = cv2.GaussianBlur(roi, (91, 91), 0)
            img[y:y + h, x:x + w] = blur
        if len(detections) == 0:
            cv2.putText(img, 'No Face Found!', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

    else:
        # Eye detection: blur the eyes and add a box around them
        for (x, y, w, h) in detections:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            roi = img[y:y + h, x:x + w]
            blur = cv2.GaussianBlur(roi, (25, 25), 0)
            img[y:y + h, x:x + w] = blur
        if len(detections) == 0:
            cv2.putText(img, 'No Eyes Found!', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

    # Display the resulting frame
    cv2.imshow('Face or Eyes Detection', img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
