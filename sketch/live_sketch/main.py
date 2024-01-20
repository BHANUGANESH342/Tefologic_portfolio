import cv2 as cv
import os
import tkinter as tk


def sketch(image, sketch_type):
    if sketch_type == "pencil":
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        img_gray_inv = 255 - img_gray
        img_blur = cv.GaussianBlur(img_gray_inv, (21, 21), 0)
        final_img = cv.divide(img_gray, 255 - img_blur, scale=256)
        return final_img
    elif sketch_type == "live":
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        img_blur = cv.GaussianBlur(img_gray, (7, 7), 0)
        canny = cv.Canny(img_blur, 60, 70)
        ret, thres = cv.threshold(canny, 90, 120, cv.THRESH_BINARY)
        return thres


def on_select(selection):
    global sketch_type
    sketch_type = selection
    root.destroy()



root = tk.Tk()
root.title("Sketch Type Selection")

label = tk.Label(root, text="Select sketch type:")
label.pack()

sketch_type = ""

pencil_button = tk.Button(root, text="Pencil Sketch", command=lambda: on_select("pencil"))
pencil_button.pack()

live_button = tk.Button(root, text="Live Sketch", command=lambda: on_select("live"))
live_button.pack()

root.mainloop()

if not sketch_type:
    exit()

# Set the output folder path
output_folder = r'C:\Users\rkssp\Desktop\virtual envi\sketch\live_sketch\livesketch\output'
os.makedirs(output_folder, exist_ok=True)

# Initializing webcam
cap = cv.VideoCapture(0)

count = 1

while True:
    ret, frame = cap.read()

    # Apply the selected sketch type
    if sketch_type == "pencil":
        sketch_img = sketch(frame, "pencil")
    else:
        sketch_img = sketch(frame, "live")

    cv.imshow("Sketch", sketch_img)

    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Press 's' to save the sketch
    elif key == ord('s'):
        sketch_filename = os.path.join(output_folder, f'{sketch_type}_sketch_{count}.jpg')
        cv.imwrite(sketch_filename, sketch_img)
        print(f"{sketch_type.capitalize()} sketch saved: {sketch_filename}")
        count += 1

cap.release()
cv.destroyAllWindows()
