import qrcode
import os
from datetime import datetime
import pyttsx3
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import cv2
import numpy as np
import webbrowser

class QRCodeApp:
    def __init__(self, master):
        self.master = master
        self.master.title("QR Code Action")

        # Add this line to create a folder for cropped QR codes
        self.crop_folder = r"C:\Users\rkssp\Desktop\virtual envi\QR\qr code\output\crop"
        os.makedirs(self.crop_folder, exist_ok=True)

        # Add this line to create an output folder for scanned QR codes
        self.output_folder = r"C:\Users\rkssp\Desktop\virtual envi\QR\qr code\output\selection"
        os.makedirs(self.output_folder, exist_ok=True)

        # Add this line to create a folder for scanned links
        self.links_folder = r"C:\Users\rkssp\Desktop\virtual envi\QR\qr code\output\links"
        os.makedirs(self.links_folder, exist_ok=True)

        tk.Label(master, text="What would you like to do?").pack()

        tk.Button(master, text="Generate QR Code", command=self.generate_qr_code).pack(pady=10)
        tk.Button(master, text="Scan QR Code (Camera)", command=self.scan_qr_code_camera).pack(pady=10)
        tk.Button(master, text="Scan Generated QR Code", command=self.scan_generated_qr_code).pack(pady=10)

        # Bind the Escape key to exit the main loop
        master.bind("<Escape>", self.exit_main_loop)

    def exit_main_loop(self, event):
        self.master.destroy()

    def generate_qr_code(self):
        try:
            qr_type, user_input = self.get_user_input()

            today_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_file_path = os.path.join(self.output_folder, f"{qr_type}_{today_date}.png")

            self.create_qr(user_input, output_file_path)
            print(f"QR code ({qr_type}) saved to: {output_file_path}")

            self.show_congratulations_popup()

            engine = pyttsx3.init()
            engine.say("Thank you! Your QR code is generated.")
            engine.runAndWait()

            # Ask if the user wants to scan the generated QR code
            answer = messagebox.askyesno("Scan Generated QR Code", "Do you want to scan the generated QR code?")
            if answer:
                # Scan the generated QR code
                self.scan_generated_qr_code(output_file_path)

        except Exception as e:
            print(f"Error generating QR code: {e}")

    def scan_qr_code_camera(self):
        try:
            # Create a video capture object (0 represents the default camera)
            cap = cv2.VideoCapture(0)

            while True:
                # Capture each frame from the camera
                ret, frame = cap.read()

                # Process the frame
                detector = cv2.QRCodeDetector()
                retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(frame)

                if retval:
                    for value in decoded_info:
                        print(f"Decoded QR code: {value}")

                        # Save scanned QR codes in a date-wise text file
                        today_date = datetime.now().strftime("%Y-%m-%d")
                        output_file_path = os.path.join(self.links_folder, f"scanned_links_{today_date}.txt")

                        with open(output_file_path, "a") as file:
                            file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {value}\n")

                        # Process scanned data based on the QR code type
                        self.process_scanned_data(value, None, points)

                        # Exit the main loop if the scanned QR code is a link
                        if self.is_link(value):
                            self.master.destroy()
                            return

                # Display the current frame
                cv2.imshow('QR Code Scanner', frame)

                # Check for the 'Esc' key to exit
                if cv2.waitKey(1) == 27:
                    break

            # Release the video capture object and close all windows
            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error scanning QR code through the camera: {e}")

    def is_link(self, data):
        # Check if the scanned QR code data is a link
        return data.startswith("http") or data.startswith("www")

    def scan_generated_qr_code(self, file_path=None):
        try:
            if file_path is None:
                # Ask the user to select an image file if not provided
                file_path = filedialog.askopenfilename(title="Select QR Code Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

                if not file_path:
                    print("No file selected. Exiting.")
                    return

            # Read the selected image
            img = cv2.imread(file_path)

            # Process the image
            detector = cv2.QRCodeDetector()
            retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(img)

            if retval:
                for value in decoded_info:
                    print(f"Decoded QR code: {value}")

                    # Process scanned data based on the QR code type
                    self.process_scanned_data(value, file_path, points)

                    # Show decoded text in a popup
                    self.show_decoded_text_popup(value)

                    # Exit the loop if needed
                    break

            # Display the image
            cv2.imshow('Scanned QR Code', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error scanning QR code through the camera: {e}")

    def process_scanned_data(self, data, file_path=None, pts=None):
        print(f"Scanned QR code data: {data}")
        if self.is_link(data):
            engine = pyttsx3.init()
            engine.say("Scanning successful. Opening the link.")
            engine.runAndWait()

            # Save and show the cropped QR code
            self.save_and_show_cropped_qr(pts)

            webbrowser.open(data)

        else:
            print("Scanned data does not match the generated QR code.")

    def save_and_show_cropped_qr(self, pts):
        try:
            img = cv2.imread(file_path)
            rect_pts = np.array(pts, dtype=int)
            rect_pts = rect_pts.reshape((-1, 1, 2))

            # Create a mask and fill it with zeros
            mask = np.zeros_like(img)

            # Fill the mask with the contour of the QR code
            cv2.drawContours(mask, [rect_pts], -1, (255, 255, 255), thickness=cv2.FILLED)

            # Bitwise-AND operation to get the cropped QR code
            cropped_qr = cv2.bitwise_and(img, mask)

            # Save the cropped QR code
            today_date = datetime.now().strftime("%Y-%m-%d")
            output_file_path = os.path.join(self.crop_folder, f"cropped_qr_{today_date}.png")
            cv2.imwrite(output_file_path, cropped_qr)
            print(f"Cropped QR code saved to: {output_file_path}")

            # Display the cropped QR code using cv2.imshow
            cv2.imshow('Cropped QR Code', cropped_qr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Decode and show the text from the cropped QR code
            decoded_text = self.decode_qr_code(output_file_path)
            self.show_decoded_text_popup(decoded_text)

            # Popup the cropped image using cv2.imshow
            cv2.imshow('Cropped Image', cropped_qr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error saving and showing cropped QR code: {e}")

    def decode_qr_code(self, image_path):
        image = cv2.imread(image_path)
        detector = cv2.QRCodeDetector()
        retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(image)
        decoded_text = decoded_info[0] if retval else "Decoding failed"
        return decoded_text

    def show_decoded_text_popup(self, decoded_text):
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Decoded Text", f"The decoded text is:\n\n{decoded_text}")

    def get_user_input(self):
        print("Choose the type of QR code:")
        print("1. Text")
        print("2. Video Link")
        print("3. Payment Information")
        print("4. Contact Information")
        print("5. Event Details")

        choice = int(simpledialog.askstring("QR Code Type", "Enter the number corresponding to your choice:"))

        if choice == 1:
            return "Text", simpledialog.askstring("Enter Text", "Enter the text to encode:")
        elif choice == 2:
            return "Video Link", simpledialog.askstring("Enter Video Link", "Enter the video link to encode:")
        elif choice == 3:
            return "Payment Information", simpledialog.askstring("Enter Payment Information", "Enter the payment information to encode:")
        elif choice == 4:
            return "Contact Information", self.get_contact_information()
        elif choice == 5:
            return "Event Details", self.get_event_details()
        else:
            print("Invalid choice. Exiting.")
            exit()

    def get_contact_information(self):
        name = simpledialog.askstring("Contact Information", "Enter the contact name:")
        number = simpledialog.askstring("Contact Information", "Enter the contact number:")

        contact_info = f"BEGIN:VCARD\nVERSION:3.0\nFN:{name}\nTEL:{number}\nEND:VCARD"
        return contact_info

    def get_event_details(self):
        event_name = simpledialog.askstring("Event Details", "Enter the name of the event:")
        additional_text = simpledialog.askstring("Event Details", "Enter additional text for the event details:")

        event_info = f"BEGIN:VEVENT\nSUMMARY:{event_name}\nDTSTART:20220101T120000\nDTEND:20220101T140000\nLOCATION:Example Venue\nDESCRIPTION:{additional_text}\nEND:VEVENT"
        return event_info

    def create_qr(self, data, output_path):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img.save(output_path)
        img.show()

    def show_congratulations_popup(self):
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Congratulations!", "Your QR code is generated.\nThank you!")

if __name__ == "__main__":
    root = tk.Tk()
    app = QRCodeApp(root)
    root.mainloop()
