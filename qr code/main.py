import cv2 
import numpy as np
from pyzbar.pyzbar import decode
import os
from datetime import datetime
import sys

output_folder = r"C:\Users\rkssp\Desktop\virtual envi\QR\qr code\output"
os.makedirs(output_folder, exist_ok=True)

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, adjust as needed

while True:
    success, img = cap.read()

    try:
        for code in decode(img):
            data = code.data.decode('utf-8')

            # Print non-ASCII characters by encoding and decoding
            print(data.encode(sys.stdout.encoding, errors='replace').decode('utf-8', 'replace'))


            # Save scanned links in a date-wise text file
            today_date = datetime.now().strftime("%Y-%m-%d")
            output_file_path = os.path.join(output_folder, f"scanned_links_{today_date}.txt")

            with open(output_file_path, "a") as file:
                file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {data}\n")

            pts = np.array([code.polygon], np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 255, 0), 5)
            font = cv2.FONT_HERSHEY_PLAIN
            font_size = 2
            cv2.putText(img, data, (code.rect[0], code.rect[1]), font, font_size, (0, 255, 0), 2)
    except Exception as e:
        print(f"Error: {e}")

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
