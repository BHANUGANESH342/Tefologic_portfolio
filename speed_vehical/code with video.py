import cv2
import dlib
import time
import math
import os
import easyocr  # Import the easyocr library

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
video = cv2.VideoCapture(0)  # 0 corresponds to the default webcam, change it if you have multiple cameras

# Constant Declaration
WIDTH = 1000
HEIGHT = 1080
OUTPUT_FOLDER = r"C:\Users\rkssp\Desktop\TEFOlogic PROJECTS\vehicle-speed-detection\OUT_PUT"
NUMBER_PLATE_FOLDER = os.path.join(OUTPUT_FOLDER, "number_plate_output")  # New folder for number plate images
FACE_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "face_output")  # New folder for face images
HELMET_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "helmet_output")  # New folder for helmet images
MASK_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "mask_output")  # New folder for mask images
SPEEDING_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "speeding_output")  # New folder for speeding vehicle images
RECORDED_VIDEO_FOLDER = os.path.join(OUTPUT_FOLDER, "recorded_video")  # New folder for recorded video

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
MAX_SPEED_LIMIT = 30


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


# Tracking multiple objects
def trackMultipleObjects():
    rectangleColor = (0, 255, 255)
    speedingCarColor = (0, 0, 255)  # Red color for the speeding car
    frameCounter = 0
    currentCarID = 0
    fps = 0

    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    out = cv2.VideoWriter(os.path.join(RECORDED_VIDEO_FOLDER, 'outTraffic.avi'),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (WIDTH, HEIGHT))

    while True:
        start_time = time.time()
        rc, image = video.read()
        if type(image) == type(None):
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
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

                        # Save the number plate image to the specified output folder
                        img_name_plate = f"number_plate_{time.time()}.png"
                        img_path_plate = os.path.join(NUMBER_PLATE_FOLDER, img_name_plate)

                        # Add timestamp to the number plate image
                        cv2.putText(imgRoi, str(time.ctime()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                                    2)

                        cv2.imwrite(img_path_plate, imgRoi)
                except Exception as e:
                    print(f"Error in OCR: {e}")

                #cv2.putText(resultImage, 'Number Plate', (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                cv2.imshow('ROI', imgRoi)

        # Face detection using Dlib
        gray = cv2.cvtColor(resultImage, cv2.COLOR_BGR2GRAY)
        faces = faceDetector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(resultImage, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Save the face image to the specified output folder
            img_name_face = f"face_{time.time()}.png"
            img_path_face = os.path.join(FACE_OUTPUT_FOLDER, img_name_face)
            cv2.imwrite(img_path_face, resultImage[y:y + h, x:x + w])

            # Crop and save the face region
            face_roi = resultImage[y:y + h, x:x + w]
            cv2.putText(face_roi, str(time.ctime()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(img_path_face, face_roi)

            # Detect helmet and mask on the driver's face
            face_gray = gray[y:y + h, x:x + w]
            resultImage[y:y + h, x:x + w] = detectHelmet(face_gray, resultImage[y:y + h, x:x + w])
            resultImage[y:y + h, x:x + w] = detectMask(face_gray, resultImage[y:y + h, x:x + w])

        # Object tracking
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
                    print(' Creating a new tracker' + str(currentCarID))

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

            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        end_time = time.time()

        if not (end_time == start_time):
            fps = 1.0 / (end_time - start_time)

        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]

                carLocation1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x1, y2, w2, h2])

                    if speed[i] is not None and y1 >= 180:
                        if speed[i] > MAX_SPEED_LIMIT:
                            # Crop and capture only the region around the speeding car
                            crop_img = resultImage[y1:y1 + h1, x1:x1 + w1]
                            # Save the cropped image to the specified output folder
                            img_name_speeding = f"speeding_car_{i}_speed_{int(speed[i])}_cropped.png"
                            img_path_speeding = os.path.join(SPEEDING_OUTPUT_FOLDER, img_name_speeding)
                            cv2.imwrite(img_path_speeding, crop_img)

                            # Draw a rectangle around the speeding car in the original frame
                            cv2.rectangle(resultImage, (x1, y1), (x1 + w1, y1 + h1), speedingCarColor, 4)

                            # Capture image of the rider
                            rider_img = resultImage[y2:y2 + h2, x2:x2 + w2]
                            img_name_rider = f"rider_{i}_speed_{int(speed[i])}.png"
                            img_path_rider = os.path.join(OUTPUT_FOLDER,
                                                          img_name_rider)  # Change to the main OUTPUT_FOLDER
                            cv2.imwrite(img_path_rider, rider_img)

                            cv2.putText(
                                resultImage,
                                str(int(speed[i])) + "km/h",
                                (int(x1 + w1 / 2), int(y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                (0, 0, 100),
                                2,
                            )

        cv2.imshow('result', resultImage)

        out.write(resultImage)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    out.release()


if __name__ == '__main__':
    trackMultipleObjects()
