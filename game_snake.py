import cv2
import mediapipe as mp
import random
import sys
import subprocess

# Initialization
def run_snake_game():
    try:
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        screen_width, screen_height = 640, 480
        cell_size, fruit_size = 20, 40
        snake_speed, smooth_factor = 1, 0.8
        min_speed_threshold = 5

        snake = [(screen_width // 2, screen_height // 2)]
        snake_direction = (0, -1)
        smoothed_vector_x, smoothed_vector_y = 0, 0
        hand_x, hand_y = 0, 0

        # Randomize fruit location
        def generate_fruit_location():
            return (random.randint(0, (screen_width - fruit_size) // cell_size) * cell_size,
                    random.randint(0, (screen_height - fruit_size) // cell_size) * cell_size)

        fruit = generate_fruit_location()

        cap = cv2.VideoCapture(0)

        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while True:
                ret, image = cap.read()
                if not ret:
                    continue

                image = cv2.flip(image, 1)
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Hand tracking
                if results.multi_hand_landmarks:
                    for landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)
                        wrist_x = int(landmarks.landmark[mp_hands.HandLandmark.WRIST].x * screen_width)
                        wrist_y = int(landmarks.landmark[mp_hands.HandLandmark.WRIST].y * screen_height)
                        cv2.circle(image, (wrist_x, wrist_y), 8, (0, 255, 0), -1)

                        # Hand gesture processing
                        prev_hand_x, prev_hand_y = hand_x, hand_y
                        hand_x, hand_y = wrist_x, wrist_y

                        vector_x = hand_x - prev_hand_x
                        vector_y = hand_y - prev_hand_y

                        smoothed_vector_x = (1 - smooth_factor) * vector_x + smooth_factor * smoothed_vector_x
                        smoothed_vector_y = (1 - smooth_factor) * vector_y + smooth_factor * smoothed_vector_y

                        # Snake direction control based on hand movement
                        if abs(smoothed_vector_x) > min_speed_threshold or abs(smoothed_vector_y) > min_speed_threshold:
                            if abs(smoothed_vector_x) > abs(smoothed_vector_y):
                                if smoothed_vector_x < 0 and snake_direction != (1, 0):
                                    snake_direction = (-1, 0)
                                elif smoothed_vector_x > 0 and snake_direction != (-1, 0):
                                    snake_direction = (1, 0)
                            else:
                                if smoothed_vector_y < 0 and snake_direction != (0, 1):
                                    snake_direction = (0, -1)
                                elif smoothed_vector_y > 0 and snake_direction != (0, -1):
                                    snake_direction = (0, 1)

                # Update snake position
                snake_head = (snake[0][0] + snake_direction[0] * cell_size, snake[0][1] + snake_direction[1] * cell_size)
                snake.insert(0, snake_head)

                # Check for collision with fruit
                if (snake_head[0] <= fruit[0] + fruit_size and snake_head[0] + cell_size >= fruit[0] and
                        snake_head[1] <= fruit[1] + fruit_size and snake_head[1] + cell_size >= fruit[1]):
                    fruit = generate_fruit_location()  # Respawn fruit
                else:
                    snake.pop()

                # Check for collision with itself
                if snake_head in snake[1:]:
                    break

                # Display snake and fruit
                for segment in snake:
                    cv2.rectangle(image, (segment[0], segment[1]), (segment[0] + cell_size, segment[1] + cell_size),
                                  (0, 0, 255), -1)

                cv2.rectangle(image, (fruit[0], fruit[1]), (fruit[0] + fruit_size, fruit[1] + fruit_size),
                              (0, 255, 0), -1)

                cv2.imshow('Snake Game', image)
                key = cv2.waitKey(1)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()
            # Запуск меню после завершения игры
            subprocess.run(["C:/PYTHON_/_PROJECT_PYTHON/Python_Project_Other/CompVision_ProjKip_2V/venv/Scripts/python.exe",
                            "menu.py"])
            sys.exit()
    except:
        cap.release()
        cv2.destroyAllWindows()
        # Запуск меню после завершения игры
        subprocess.run(["C:/PYTHON_/_PROJECT_PYTHON/Python_Project_Other/CompVision_ProjKip_2V/venv/Scripts/python.exe",
                        "menu.py"])
        sys.exit()


