import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
import time
import pyttsx3
import tkinter as tk
from tkinter import simpledialog

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to set alert duration
def set_alert_duration():
    global alert_duration
    alert_duration = simpledialog.askinteger("Input", "Enter the duration in seconds for the alert:")

# Function to set cooldown duration
def set_cooldown_duration():
    global cooldown_duration
    cooldown_duration = simpledialog.askinteger("Input", "Enter the cooldown duration in seconds for the alert:")

# Function to set alert message
def set_alert_message():
    global text_to_speech
    text_to_speech = simpledialog.askstring("Input", "Enter the alert message:")

# Initialize Tkinter window
root = tk.Tk()
root.title("Blink Detection System")

# Buttons to set parameters
alert_button = tk.Button(root, text="Set Alert Duration", command=set_alert_duration)
alert_button.pack(pady=10)

cooldown_button = tk.Button(root, text="Set Cooldown Duration", command=set_cooldown_duration)
cooldown_button.pack(pady=10)

message_button = tk.Button(root, text="Set Alert Message", command=set_alert_message)
message_button.pack(pady=10)

# Close window button
def close_window():
    global root
    root.destroy()

start_button = tk.Button(root, text="Start Blink Detection", command=close_window)
start_button.pack(pady=20)

root.mainloop()

# Load Haar cascades and dlib predictor
face_cascade = cv2.CascadeClassifier(r"C:\Users\rkssp\Desktop\MAIN TELFLOGIC\main project\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"C:\Users\rkssp\Downloads\haarcascade_mcs_eyepair_big.xml")
predictor = dlib.shape_predictor(r"C:\Users\rkssp\Downloads\facial-landmarks-recognition-master\facial-landmarks-recognition-master\shape_predictor_68_face_landmarks.dat")

# Initialize video capture, detector, and speech engine
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
engine = pyttsx3.init()

# Initialize parameters
count = 0
total = 0
last_eye_detection_time = time.time()
audio_played = False
last_alert_time = 0
faces_detected = False

# Set EAR threshold for eye closure detection
EAR_THRESHOLD = 0.2

# Main loop
while True:
    success, img = cap.read()
    if not success:
        break

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        faces_detected = True
    else:
        faces_detected = False
        count = 0
        total = 0
        audio_played = False
        last_alert_time = 0

    eyes_detected = False

    if faces_detected:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                eyes_detected = True
                last_eye_detection_time = time.time()

        faces = detector(img_gray)
        for face in faces:
            landmarks = predictor(img_gray, face)
            landmarks = face_utils.shape_to_np(landmarks)
            left_eye = landmarks[42:48]
            right_eye = landmarks[36:42]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            if ear < EAR_THRESHOLD:
                count += 1
            else:
                if count >= 3:
                    total += 1
                count = 0

        if not eyes_detected:
            cv2.putText(img, "Error: Eyes not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elapsed_time = int(time.time() - last_eye_detection_time)
            cv2.putText(img, "Time: {} sec".format(elapsed_time), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if elapsed_time >= alert_duration and not audio_played:
                engine.say(text_to_speech)
                engine.runAndWait()
                audio_played = True
                last_alert_time = time.time()
        else:
            audio_played = False
            count = 0

        if audio_played and time.time() - last_alert_time >= cooldown_duration:
            engine.say(text_to_speech)
            engine.runAndWait()
            last_alert_time = time.time()

    cv2.putText(img, "Blink Count: {}".format(total), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    watermark = "I am watching you"
    (text_width, text_height), _ = cv2.getTextSize(watermark, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = int(img.shape[1] / 2 - text_width / 2)
    text_y = int(img.shape[0] - text_height)
    cv2.putText(img, watermark, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
