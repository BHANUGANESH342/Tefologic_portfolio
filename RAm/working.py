

import cv2
import os
from gtts import gTTS
import pygame
import numpy as np
from io import BytesIO
from PIL import ImageFont, ImageDraw, Image

# Initialize the pygame mixer
pygame.mixer.init()


def select_language():
    # Assuming Telugu is selected by default
    return 'te'


# Set language to Telugu
selected_language = select_language()

image_folder = r"C:\Users\rkssp\Desktop\virtual envi\JAI_SRI_RAM\RAm\loard_rama"  # Replace with the actual path to your image folder

# Path to the Telugu font file (change this to the correct path)
telugu_font_path = "Pothana2000.ttf"

exit_key = 27  # ASCII code for the Esc key


def play_telugu_audio(text_to_speak):
    tts = gTTS(text=text_to_speak, lang='te')
    audio_stream = BytesIO()
    tts.write_to_fp(audio_stream)
    audio_stream.seek(0)

    pygame.mixer.music.load(audio_stream)
    pygame.mixer.music.play()


def on_button_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if 50 < x < 200 and 150 < y < 200:
            open_fullscreen_window()
        elif 250 < x < 400 and 150 < y < 200:
            print("Button 2 clicked")

            # Call the function to perform iterations with a different mantra
            perform_iterations_with_custom_mantra()


def perform_iterations_with_custom_mantra():
    custom_mantra = input(f"Enter the custom mantra in Telugu: ")
    images = [os.path.join(image_folder, image) for image in sorted(os.listdir(image_folder))]
    window_name = 'Custom Mantra Window'

    # Perform iterations with the custom mantra
    perform_iterations(images, window_name, custom_mantra)


def perform_iterations(images, window_name, mantra="శ్రీ రామ జయ రామ జయ జయ రామ్", total_iterations=108):
    iterations_done = 0

    while iterations_done < total_iterations:
        for image_path in images:
            # Speak the text
            text_to_speak = mantra
            play_telugu_audio(text_to_speak)

            # Display the image within the specified window
            img = cv2.imread(image_path)

            # Render Telugu text on the image
            img = render_telugu_text(img, text_to_speak)

            # Display iteration count as text at the bottom
            iteration_text = f'Iteration: {iterations_done + 1}/{total_iterations}'
            img = render_telugu_text(img, iteration_text, position=(50, 470), font_size=0.5)

            cv2.imshow(window_name, cv2.resize(img, (800, 500)))  # Resize the window for display

            # Add a delay (adjust as needed)
            key = cv2.waitKey(2000)  # Display the image for 2 seconds

            if key == exit_key:  # Check for the "Esc" key
                cv2.destroyAllWindows()
                return

            # Check if music is still playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

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
    cv2.putText(popup, f'Taraka_mantra or custom ({selected_language})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

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


def render_telugu_text(img, text, position=(50, 50), font_size=1.0):
    # Convert BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(img_rgb)

    # Load Telugu font
    telugu_font = ImageFont.truetype(telugu_font_path, int(24 * font_size))

    # Create ImageDraw object
    draw = ImageDraw.Draw(pil_img)

    # Render Telugu text on the image
    draw.text(position, text, font=telugu_font, fill=(255, 255, 255))

    # Convert PIL Image back to numpy array
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return img_bgr


# Call the function to show the popup
show_popup()

# Call the function to show the popup
show_popup()
