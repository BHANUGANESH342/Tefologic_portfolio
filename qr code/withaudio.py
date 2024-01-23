import qrcode
import os
from datetime import datetime
import pyttsx3

def create_qr(data, output_path):
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

def get_text_input(prompt):
    return input(prompt)

def get_contact_information():
    name = get_text_input("Enter the contact name: ")
    number = get_text_input("Enter the contact number: ")

    contact_info = f"BEGIN:VCARD\nVERSION:3.0\nFN:{name}\nTEL:{number}\nEND:VCARD"
    return contact_info

def get_event_details():
    event_name = get_text_input("Enter the name of the event: ")
    additional_text = get_text_input("Enter additional text for the event details: ")

    event_info = f"BEGIN:VEVENT\nSUMMARY:{event_name}\nDTSTART:20220101T120000\nDTEND:20220101T140000\nLOCATION:Example Venue\nDESCRIPTION:{additional_text}\nEND:VEVENT"
    return event_info

def get_user_input():
    print("Choose the type of QR code:")
    print("1. Text")
    print("2. Video Link")
    print("3. Payment Information")
    print("4. Contact Information")
    print("5. Event Details")

    choice = int(get_text_input("Enter the number corresponding to your choice: "))

    if choice == 1:
        return "Text", get_text_input("Enter the text to encode: ")  # Example: Hello, this is an example text.
    elif choice == 2:
        return "Video Link", get_text_input("Enter the video link to encode: ")  # Example: https://www.example.com/sample_video.mp4
    elif choice == 3:
        return "Payment Information", get_text_input("Enter the payment information to encode: ")  # Example: bitcoin:1exampleaddress?amount=0.001&label=Example%20Merchant
    elif choice == 4:
        return "Contact Information", get_contact_information()
    elif choice == 5:
        return "Event Details", get_event_details()
    else:
        print("Invalid choice. Exiting.")
        exit()

def main():
    qr_type, user_input = get_user_input()

    output_folder = r"C:\Users\rkssp\Desktop\virtual envi\QR\qr code\output\selection"
    os.makedirs(output_folder, exist_ok=True)

    today_date = datetime.now().strftime("%Y-%m-%d")
    output_file_path = os.path.join(output_folder, f"{qr_type}_{today_date}.png")

    create_qr(user_input, output_file_path)
    print(f"QR code ({qr_type}) saved to: {output_file_path}")

    # Text-to-speech for thanking the user
    engine = pyttsx3.init()
    engine.say("Thank you! Your QR code is generated.")
    engine.runAndWait()

if __name__ == "__main__":
    main()
