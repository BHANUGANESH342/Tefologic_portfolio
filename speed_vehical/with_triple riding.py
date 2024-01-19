import cv2
import dlib
import time
import math
import os
import easyocr
import numpy as np
import easygui

# Load YOLO model and configuration
net = cv2.dnn.readNet(r"C:\Users\rkssp\Downloads\yolov3.weights", r"C:\Users\rkssp\Downloads\yolov3 (1).cfg")

with open(r"C:\Users\rkssp\Downloads\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getUnconnectedOutLayersNames()

# Cascade file paths
carCascade = cv2.CascadeClassifier("vech.xml")
noPlateCascadePath = r"C:\Users\rkssp\Desktop\TEFOlogic PROJECTS\number plate2\indian_license_plate.xml"
noPlateCascade = cv2.CascadeClassifier(noPlateCascadePath)
faceCascadePath = r"C:\Users\rkssp\Desktop\TEFOlogic PROJECTS\face blur\haarcascade_frontalface_default.xml"
helmetCascadePath = r"C:\Users\rkssp\Downloads\haarcascade_helmet.xml"
maskCascadePath = r"C:\Users\rkssp\Downloads\haarcascade_mcs_nose.xml"

# Load the Dlib face detector
faceDetector = dlib.get_frontal_face_detector()

# Load the helmet and mask cascades
helmetCascade = cv2.CascadeClassifier(helmetCascadePath)
maskCascade = cv2.CascadeClassifier(maskCascadePath)

# Webcam video capture
video = cv2.VideoCapture(r"C:\Users\rkssp\Desktop\portfolio\triple-riding.jpg")

# Constant Declaration
WIDTH = 1000
HEIGHT = 1080
OUTPUT_FOLDER = r"C:\Users\rkssp\Desktop\virtual envi\MAIN_PROJECT\main project\main project\OUT_PUT"
NUMBER_PLATE_FOLDER = os.path.join(OUTPUT_FOLDER, "number_plate_output")
FACE_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "face_output")
HELMET_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "helmet_output")
MASK_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "mask_output")
SPEEDING_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "speeding_output")
RECORDED_VIDEO_FOLDER = os.path.join(OUTPUT_FOLDER, "recorded_video")

# Create output folders if they don't exist
os.makedirs(NUMBER_PLATE_FOLDER, exist_ok=True)
os.makedirs(FACE_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(HELMET_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MASK_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SPEEDING_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(RECORDED_VIDEO_FOLDER, exist_ok=True)

# Initialize the easyocr reader with the 'en' language
reader = easyocr.Reader(['en'])

# Maximum Speed Limit
MAX_SPEED_LIMIT = 20


# Estimate speed function
def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8.8
    d_meters = d_pixels / ppm
    fps = 18
    speed = d_meters * fps * 3.6
    return speed


# Detect helmet on the face region
def detectHelmet(face_gray, face_roi):
    helmets = helmetCascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (hx, hy, hw, hh) in helmets:
        cv2.rectangle(face_roi, (hx, hy), (hx + hw, hy + hh), (0, 0, 255), 2)
        helmet_img = face_roi[hy:hy + hh, hx:hx + hw]
        img_name_helmet = f"helmet_{time.time()}.png"
        img_path_helmet = os.path.join(HELMET_OUTPUT_FOLDER, img_name_helmet)
        cv2.imwrite(img_path_helmet, helmet_img)
    return face_roi


# Detect mask on the face region
def detectMask(face_gray, face_roi):
    masks = maskCascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (mx, my, mw, mh) in masks:
        cv2.rectangle(face_roi, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)
        mask_img = face_roi[my:my + mh, mx:mx + mw]
        img_name_mask = f"mask_{time.time()}.png"
        img_path_mask = os.path.join(MASK_OUTPUT_FOLDER, img_name_mask)
        cv2.imwrite(img_path_mask, mask_img)
    return face_roi


# Triple Riding Detection Function
def tripleRidingDetection(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Information to be extracted
    class_ids = []
    confidences = []
    boxes = []

    # Analyze detected objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class ID 0 is for a person in COCO dataset
                # Object detected is a person
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove redundant boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and count persons
    persons_count = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            persons_count += 1

    # Display the count on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Persons Count: {persons_count}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Triple riding detection
    if persons_count == 3:
        message = "Triple riding detected!"
        easygui.msgbox(message, "Triple Riding Alert")
        print(message)

    return frame


# Tracking multiple objects
def trackMultipleObjects():
    rectangleColor = (0, 255, 255)
    speedingCarColor = (0, 0, 255)
    frameCounter = 0
    currentCarID = 0
    fps = 0

    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    out = cv2.VideoWriter(
        os.path.join(RECORDED_VIDEO_FOLDER, 'outTraffic.avi'),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        10,
        (WIDTH, HEIGHT)
    )

    max_speed = 20

    while True:
        start_time = time.time()
        rc, image = video.read()
        if type(image) == type(None):
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))

        # Call the tripleRidingDetection function
        image = tripleRidingDetection(image)

        resultImage = image.copy()

        frameCounter = frameCounter + 1
        carIDtoDelete = []

        # Number plate detection
        imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        nPlates = noPlateCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=3)

        for (x, y, w, h) in nPlates:
            area = w * h
            if area > 500:
                cv2.rectangle(resultImage, (x, y), (x + w, y + h), (0, 255, 0), 4)
                imgRoi = resultImage[y:y + h, x:x + w]

                try:
                    # Perform OCR on the number plate region using easyocr
                    result = reader.readtext(imgRoi)
                    if result:
                        numberPlate = result[0][-1]
                        print("Number Plate:", numberPlate)

                        img_name_plate = f"number_plate_{time.time()}.png"
                        img_path_plate = os.path.join(NUMBER_PLATE_FOLDER, img_name_plate)

                        cv2.putText(imgRoi, str(time.ctime()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                        cv2.imwrite(img_path_plate, imgRoi)
                except Exception as e:
                    print(f"Error in OCR: {e}")

        # Face detection using Dlib
        gray = cv2.cvtColor(resultImage, cv2.COLOR_BGR2GRAY)
        faces = faceDetector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(resultImage, (x, y), (x + w, y + h), (255, 0, 0), 2)

            img_name_face = f"face_{time.time()}.png"
            img_path_face = os.path.join(FACE_OUTPUT_FOLDER, img_name_face)
            cv2.imwrite(img_path_face, resultImage[y:y + h, x:x + w])

            face_roi = resultImage[y:y + h, x:x + w]
            cv2.putText(face_roi, str(time.ctime()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(img_path_face, face_roi)

            face_gray = gray[y:y + h, x:x + w]
            resultImage[y:y + h, x:x + w] = detectHelmet(face_gray, resultImage[y:y + h, x:x + w])
            resultImage[y:y + h, x:x + w] = detectMask(face_gray, resultImage[y:y + h, x:x + w])

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(resultImage)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            print("Removing carID " + str(carID) + ' from the list of trackers.')
            print("Removing carID " + str(carID) + ' previous location.')
            print("Removing carID " + str(carID) + ' current location.')
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        if not (frameCounter % 10):
            gray = cv2.cvtColor(resultImage, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if (
                        (t_x <= x_bar <= (t_x + t_w))
                        and (t_y <= y_bar <= (t_y + t_h))
                        and (x <= t_x_bar <= (x + w))
                        and (y <= t_y_bar <= (y + h))
                    ):
                        matchCarID = carID

                if matchCarID is None:
                    print("Creating new tracker " + str(currentCarID))

                    tracker = dlib.correlation_tracker()
                    tracker.start_track(resultImage, dlib.rectangle(x, y, x + w, y + h))

                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]

                    currentCarID = currentCarID + 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 2)

            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        if frameCounter > 1:
            end_time = time.time()
            fps = round(1 / (end_time - start_time), 2)

        if len(carLocation1) != 0 and len(carLocation2) != 0:
            for i in carLocation1.keys():
                if frameCounter % 1 == 0:
                    [x, y, w, h] = carLocation1[i]
                    [t_x, t_y, t_w, t_h] = carLocation2[i]

                    # Check if the object has moved horizontally
                    if not (
                            (x <= t_x <= x + w)
                            or (t_x <= x <= t_x + t_w)
                            or (y <= t_y <= y + h)
                            or (t_y <= y <= t_y + t_h)
                    ):
                        # Update speed estimate
                        speed[i] = estimateSpeed([x, y], [t_x, t_y])

                    # Draw the tracking rectangle
                    cv2.line(resultImage, (int(x + 0.5 * w), int(y + 0.5 * h)),
                             (int(t_x + 0.5 * t_w), int(t_y + 0.5 * t_h)),
                             rectangleColor, 2)
                    if speed[i] is not None:
                        # Display the speed
                        cv2.putText(resultImage, str(round(speed[i], 2)) + " km/h",
                                    (int(x + 0.5 * w), int(y + 0.5 * h) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                    (0, 0, 255), 2)

                        # Check for speeding
                        if speed[i] > MAX_SPEED_LIMIT:
                            message = f"Speeding detected! Speed: {round(speed[i], 2)} km/h"
                            easygui.msgbox(message, "Speeding Alert")
                            print(message)

        cv2.putText(resultImage, f'FPS: {fps}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Display the resulting frame
        cv2.imshow('Tracking', resultImage)
        out.write(resultImage)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(0) == ord('q'):
            break

        # Release video capture and writer objects
    video.release()
    out.release()

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

# Call the function to track multiple objects
trackMultipleObjects()