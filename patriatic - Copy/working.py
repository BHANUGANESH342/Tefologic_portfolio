import cv2
import numpy as np
import pygame
import sys
import random
import mediapipe as mp

# Initialize Pygame
pygame.init()

# Set screen dimensions
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Gesture Racing Game")

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
BLACK = (0, 0, 0)  # Define BLACK color

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
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)  # Use the default camera

# Define hand positions
HAND_LEFT = 'left'
HAND_RIGHT = 'right'
HAND_UP = 'up'
HAND_DOWN = 'down'

# Function to check hand position
def check_hand_position(hand_landmarks):
    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            # Get the landmarks for the hand
            landmarks = hand_landmark.landmark
            # Extract the x and y coordinates for the landmarks
            x_values = [landmark.x for landmark in landmarks]
            y_values = [landmark.y for landmark in landmarks]
            # Calculate the center of the hand
            center_x = (max(x_values) + min(x_values)) / 2
            center_y = (max(y_values) + min(y_values)) / 2
            # Determine hand position based on the center
            if center_x < 0.4:
                return HAND_LEFT
            elif center_x > 0.6:
                return HAND_RIGHT
            elif center_y < 0.4:
                return HAND_UP
            elif center_y > 0.6:
                return HAND_DOWN
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
while running:
    screen.blit(background_image, (0, 0))  # Set background image

    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(rgb_frame)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # Process hand landmarks and gestures
    hand_position = check_hand_position(results.multi_hand_landmarks)

    # Control the car based on hand position
    if hand_position == HAND_LEFT:
        car_x -= car_speed
        if car_x < 0:  # Ensure the car stays within the left boundary
            car_x = 0
    elif hand_position == HAND_RIGHT:
        car_x += car_speed
        if car_x > screen_width - car_width:  # Ensure the car stays within the right boundary
            car_x = screen_width - car_width
    elif hand_position == HAND_UP:
        car_y -= car_speed
        if car_y < 0:  # Ensure the car stays within the top boundary
            car_y = 0
    elif hand_position == HAND_DOWN:
        car_y += car_speed
        if car_y > screen_height - car_height:  # Ensure the car stays within the bottom boundary
            car_y = screen_height - car_height

    # Create new balls and coins randomly
    if random.randint(0, 100) < 7:  # Increased frequency of blue balls
        blue_balls.append(create_blue_ball())
    if random.randint(0, 100) < 5:  # Increased frequency of red balls
        red_balls.append(create_red_ball())
    if random.randint(0, 1000) < 1:  # Reduced frequency of coins
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
