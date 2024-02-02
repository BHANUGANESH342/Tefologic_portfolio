import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
import time
import pyttsx3

# Load the Haar cascade classifiers for face and eye pair detection
face_cascade = cv2.CascadeClassifier(r"C:\Users\rkssp\Desktop\MAIN TELFLOGIC\main project\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"C:\Users\rkssp\Downloads\haarcascade_mcs_eyepair_big.xml")

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\rkssp\Downloads\facial-landmarks-recognition-master\facial-landmarks-recognition-master\shape_predictor_68_face_landmarks.dat")

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Define a function to compute the eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize variables for blink counting and time tracking
count = 0
total = 0
last_eye_detection_time = time.time()
audio_played = False
last_alert_time = 0
faces_detected = False  # Flag to track if faces are detected

# Main loop for video processing
while True:
    # Read a frame from the video capture
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar cascade classifier
    faces = face_cascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if faces are detected
    if len(faces) > 0:
        faces_detected = True
    else:
        faces_detected = False
        # Reset variables and flags
        count = 0
        total = 0
        audio_played = False
        last_alert_time = 0

    # Flag to check if eyes are detected
    eyes_detected = False

    if faces_detected:
        # Iterate over detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = imgGray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Detect eyes within the face region using the Haar cascade classifier
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                eyes_detected = True  # Set to True if eyes are detected
                last_eye_detection_time = time.time()

        # Perform blink detection using dlib
        faces = detector(imgGray)
        for face in faces:
            landmarks = predictor(imgGray, face)
            landmarks = face_utils.shape_to_np(landmarks)
            leftEye = landmarks[42:48]
            rightEye = landmarks[36:42]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            if ear < 0.3:
                count += 1
            else:
                if count >= 3:
                    total += 1
                count = 0

        # Display error message if eyes are not detected
        if not eyes_detected:
            cv2.putText(img, "Error: Eyes not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Calculate the time since eyes were last detected
            elapsed_time = int(time.time() - last_eye_detection_time)
            cv2.putText(img, "Time: {} sec".format(elapsed_time), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Check if it's been 10 seconds since the last eye detection
            if elapsed_time >= 10 and not audio_played:
                # Convert text to speech
                text_to_speech = "Please wake up"
                engine.say(text_to_speech)
                engine.runAndWait()
                audio_played = True
                last_alert_time = time.time()

        else:
            # Reset the audio played flag when eyes are detected
            audio_played = False
            count = 0

        # Check if it's been 10 seconds since the last alert and 5 seconds since the last alert was played
        if audio_played and time.time() - last_alert_time >= 5:
            engine.say(text_to_speech)
            engine.runAndWait()
            last_alert_time = time.time()

    # Display blink count on the video frame
    cv2.putText(img, "Blink Count: {}".format(total), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Add movable watermark
    watermark = "I am watching you"
    (text_width, text_height), _ = cv2.getTextSize(watermark, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = int(img.shape[1] / 2 - text_width / 2)
    text_y = int(img.shape[0] - text_height)
    cv2.putText(img, watermark, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Video', img)

    # Check for 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
