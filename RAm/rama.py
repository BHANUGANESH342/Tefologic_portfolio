import cv2
import os
import pyttsx3
import numpy as np

def select_language():
    print("Available languages:")
    print("1. English (en)")
    print("2. Telugu (te)")

    language_code = input("Enter the code of the language you want to use: ").lower()

    if language_code == 'en':
        return 'en'
    elif language_code == 'te':
        return 'te'
    else:
        print("Invalid language code. Using English as the default language.")
        return 'en'

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set language based on user input
selected_language = select_language()
engine.setProperty('language', selected_language)

# Print details of available voices
voices = engine.getProperty('voices')
for idx, voice in enumerate(voices):
    print(f"Voice {idx} - ID: {voice.id}, Name: {voice.name}, Lang: {voice.languages}")

# Allow the user to select a voice index
selected_voice_index = int(input("Enter the index of the voice you want to use: "))

# Set the voice to the selected index
if 0 <= selected_voice_index < len(voices):
    engine.setProperty('voice', voices[selected_voice_index].id)
    print(f"Selected voice: {voices[selected_voice_index].name}")
else:
    print("Invalid voice index. Using the default voice.")

image_folder = r"C:\Users\rkssp\Desktop\virtual envi\JAI_SRI_RAM\RAm\loard_rama"  # Replace with the actual path to your image folder

exit_key = 27  # ASCII code for the Esc key

def on_button_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if 50 < x < 200 and 150 < y < 200:
            open_fullscreen_window()
        elif 250 < x < 400 and 150 < y < 200:
            print("Button 2 clicked")

            # Call the function to perform iterations with a different mantra
            perform_iterations_with_custom_mantra()

def perform_iterations_with_custom_mantra():
    custom_mantra = input(f"Enter the custom mantra in {selected_language}: ")
    images = [os.path.join(image_folder, image) for image in sorted(os.listdir(image_folder))]
    window_name = 'Custom Mantra Window'

    # Perform iterations with the custom mantra
    perform_iterations(images, window_name, custom_mantra)

def perform_iterations(images, window_name, mantra="Shri Rama Jaya Rama Jaya Jaya Rama", total_iterations=108):
    iterations_done = 0

    while iterations_done < total_iterations:
        for image_path in images:
            # Speak the text
            text_to_speak = mantra
            engine.say(text_to_speak)
            engine.runAndWait()

            # Display the image within the specified window
            img = cv2.imread(image_path)
            cv2.putText(img, text_to_speak, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display iteration count as text at the bottom
            iteration_text = f'Iteration: {iterations_done + 1}/{total_iterations}'
            cv2.putText(img, iteration_text, (50, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(window_name, cv2.resize(img, (800, 500)))  # Resize the window for display

            # Add a delay (adjust as needed)
            key = cv2.waitKey(2000)  # Display the image for 2 seconds

            if key == exit_key:  # Check for the "Esc" key
                cv2.destroyAllWindows()
                return

            iterations_done += 1
            if iterations_done >= total_iterations:
                break

    cv2.destroyAllWindows()

def open_fullscreen_window():
    images = [os.path.join(image_folder, image) for image in sorted(os.listdir(image_folder))]
    fullscreen_window = np.zeros((500, 800, 3), dtype=np.uint8)
    cv2.putText(fullscreen_window, 'Fullscreen Window', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.imshow('Fullscreen Window', fullscreen_window)

    # Perform iterations in the fullscreen window with the default mantra
    perform_iterations(images, 'Fullscreen Window')

    cv2.destroyAllWindows()

def show_popup():
    # Create a blank image (black background)
    popup = np.zeros((300, 500, 3), dtype=np.uint8)

    # Display text and buttons on the popup
    cv2.putText(popup, f'Taraka_mantra or custom ({selected_language})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Button 1
    cv2.rectangle(popup, (50, 150), (200, 200), (0, 255, 0), -1)
    cv2.putText(popup, 'Taraka Mantra', (70, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Button 2
    cv2.rectangle(popup, (250, 150), (400, 200), (0, 255, 0), -1)
    cv2.putText(popup, 'USER_input', (270, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Display the popup
    cv2.imshow('Taraka_mantra or custom', popup)
    cv2.setMouseCallback('Taraka_mantra or custom', on_button_click)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function to show the popup
show_popup()
