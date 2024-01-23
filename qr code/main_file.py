import qrcode
import os
from datetime import datetime
import pyttsx3
import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import webbrowser
import sys

class QRCodeApp:
    def __init__(self, master):
        self.master = master
        self.master.title("QR Code Action")

        tk.Label(master, text="What would you like to do?").pack()

        tk.Button(master, text="Generate QR Code", command=self.generate_qr_code).pack(pady=10)
        tk.Button(master, text="Scan QR Code", command=self.scan_qr_code).pack(pady=10)

    def generate_qr_code(self):
        self.master.destroy()
        try:
            qr_type, user_input = self.get_user_input()

            output_folder = r"C:\Users\rkssp\Desktop\virtual envi\QR\qr code\output\selection"
            os.makedirs(output_folder, exist_ok=True)

            today_date = datetime.now().strftime("%Y-%m-%d")
            output_file_path = os.path.join(output_folder, f"{qr_type}_{today_date}.png")

            self.create_qr(user_input, output_file_path)
            print(f"QR code ({qr_type}) saved to: {output_file_path}")

            self.show_congratulations_popup()

            engine = pyttsx3.init()
            engine.say("Thank you! Your QR code is generated.")
            engine.runAndWait()
        except Exception as e:
            print(f"Error generating QR code: {e}")

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

    def scan_qr_code(self):
        self.master.destroy()
        try:
            output_folder = r"C:\Users\rkssp\Desktop\virtual envi\QR\qr code\output"
            os.makedirs(output_folder, exist_ok=True)

            cap = cv2.VideoCapture(0)

            while True:
                success, img = cap.read()

                try:
                    for code in decode(img):
                        data = code.data.decode('utf-8')

                        print(data.encode(sys.stdout.encoding, errors='replace').decode('utf-8', 'replace'))

                        # Save scanned QR codes in a date-wise text file
                        today_date = datetime.now().strftime("%Y-%m-%d")
                        output_file_path = os.path.join(output_folder, f"scanned_qrs_{today_date}.txt")

                        with open(output_file_path, "a") as file:
                            file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {data}\n")

                        # Open the link in a web browser
                        webbrowser.open(data)

                        # Play an audio message
                        engine = pyttsx3.init()
                        engine.say(f"Opening link: {data}")
                        engine.runAndWait()

                        pts = np.array([code.polygon], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [pts], True, (0, 255, 0), 5)
                        font = cv2.FONT_HERSHEY_PLAIN
                        font_size = 2
                        cv2.putText(img, data, (code.rect[0], code.rect[1]), font, font_size, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error scanning QR code: {e}")

                cv2.imshow('Image', img)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error initiating QR code scanning: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = QRCodeApp(root)
    root.mainloop()

# upi://pay?pa=9398871369@upi&mc=your-merchant-code&tid=your-transaction-id&tr=your-transaction-reference-id&tn=your-transaction-note&am=200&cu=INR&url=your-transaction-url
