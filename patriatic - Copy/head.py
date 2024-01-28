# Import necessary libraries
import cv2
import numpy as np
import pygame
import sys
import random
import mediapipe as mp

# Initialize Pygame
pygame.init()

# Set screen dimensions
screen_width = 1200
screen_height = 800
webcam_width = 160
webcam_height = 120

# Set screen dimensions
screen = pygame.display.set_mode((screen_width, screen_height))

# Create a surface for the webcam feed
webcam_surface = pygame.Surface((webcam_width, webcam_height))

# Load background image
background_image = pygame.image.load(r"C:\Users\rkssp\Downloads\background_car (1).jpg").convert()

# Load car image
car_image = pygame.image.load(r"C:\Users\rkssp\Downloads\car (2).jpg").convert_alpha()

# Load coin image
coin_image = pygame.image.load(r"C:\Users\rkssp\Downloads\internet (2).jpg").convert_alpha()

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Car attributes
car_width = car_image.get_width()
car_height = car_image.get_height()
car_x = screen_width // 2 - car_width // 2
car_y = screen_height // 2 - car_height // 2
car_speed = 5

# Ball attributes
ball_radius = 20
ball_speed = 3
blue_balls = []
red_balls = []
coins = []

# Score
score = 0
font = pygame.font.Font(None, 36)

# OpenCV settings
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)  # Use the default camera

# Define head positions
HEAD_LEFT = 'left'
HEAD_RIGHT = 'right'
HEAD_UP = 'up'
HEAD_DOWN = 'down'

# Initialize webcam position
webcam_x = screen_width - webcam_width
webcam_y = 0

# Function to check head position
def check_head_position(face_detections):
    if face_detections.detections:
        for detection in face_detections.detections:
            bboxC = detection.location_data.relative_bounding_box
            cx = int(bboxC.xmin * screen_width + bboxC.width * screen_width / 2)
            cy = int(bboxC.ymin * screen_height + bboxC.height * screen_height / 2)

            if cx < 0.4 * screen_width:
                return HEAD_LEFT
            elif cx > 0.6 * screen_width:
                return HEAD_RIGHT
            elif cy < 0.4 * screen_height:
                return HEAD_UP
            elif cy > 0.6 * screen_height:
                return HEAD_DOWN
    return None

# Function to create a new blue ball
def create_blue_ball():
    ball_x = screen_width
    ball_y = random.randint(ball_radius, screen_height - ball_radius)
    return {'x': ball_x, 'y': ball_y}

# Function to create a new red ball
def create_red_ball():
    ball_x = screen_width
    ball_y = random.randint(ball_radius, screen_height - ball_radius)
    return {'x': ball_x, 'y': ball_y}

# Function to create a new coin
def create_coin():
    coin_x = screen_width
    coin_y = random.randint(ball_radius, screen_height - ball_radius)
    return {'x': coin_x, 'y': coin_y}

# Main game loop
running = True
clock = pygame.time.Clock()
dragging = False
offset_x = 0
offset_y = 0

while running:
    screen.blit(background_image, (0, 0))  # Set background image

    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face in the frame
    face_results = face_detection.process(rgb_frame)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if webcam_x < mouse_x < webcam_x + webcam_width and webcam_y < mouse_y < webcam_y + webcam_height:
                dragging = True
                offset_x = mouse_x - webcam_x
                offset_y = mouse_y - webcam_y
        elif event.type == pygame.MOUSEBUTTONUP:
            dragging = False
            webcam_x, webcam_y = pygame.mouse.get_pos()

    # Process face landmarks and gestures
    head_position = check_head_position(face_results)

    # Control the car based on head position
    if head_position == HEAD_LEFT:
        car_x -= car_speed
        if car_x < 0:
            car_x = 0
    elif head_position == HEAD_RIGHT:
        car_x += car_speed
        if car_x > screen_width - car_width:
            car_x = screen_width - car_width
    elif head_position == HEAD_UP:
        car_y -= car_speed
        if car_y < 0:
            car_y = 0
    elif head_position == HEAD_DOWN:
        car_y += car_speed
        if car_y > screen_height - car_height:
            car_y = screen_height - car_height

    # Create new balls and coins randomly
    if random.randint(0, 100) < 7:
        blue_balls.append(create_blue_ball())
    if random.randint(0, 100) < 5:
        red_balls.append(create_red_ball())
    if random.randint(0, 1000) < 1:
        coins.append(create_coin())

    # Move and draw blue balls
    for ball in blue_balls:
        ball['x'] -= ball_speed
        pygame.draw.circle(screen, BLUE, (ball['x'], ball['y']), ball_radius)

    # Move and draw red balls
    for ball in red_balls:
        ball['x'] -= ball_speed
        pygame.draw.circle(screen, RED, (ball['x'], ball['y']), ball_radius)

    # Move and draw coins
    for coin in coins:
        coin['x'] -= ball_speed
        screen.blit(coin_image, (coin['x'], coin['y']))

    # Check for collisions and update score
    for ball in blue_balls:
        if car_x < ball['x'] < car_x + car_width and car_y < ball['y'] < car_y + car_height:
            score += 5
            blue_balls.remove(ball)
    for ball in red_balls:
        if car_x < ball['x'] < car_x + car_width and car_y < ball['y'] < car_y + car_height:
            score -= 10
            red_balls.remove(ball)
    for coin in coins:
        if car_x < coin['x'] < car_x + car_width and car_y < coin['y'] < car_y + car_height:
            score += 20
            coins.remove(coin)

    # Draw the car image
    screen.blit(car_image, (car_x, car_y))

    # Draw score
    score_text = font.render("Score: " + str(score), True, GREEN)
    screen.blit(score_text, (10, 10))

    # Update the webcam surface with the current frame
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)

    # Update webcam position if dragging
    if dragging:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        webcam_x = mouse_x - offset_x
        webcam_y = mouse_y - offset_y

    webcam_surface.blit(frame, (0, 0))
    screen.blit(webcam_surface, (webcam_x, webcam_y))

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    clock.tick(30)

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

# Quit Pygame
pygame.quit()
sys.exit()
