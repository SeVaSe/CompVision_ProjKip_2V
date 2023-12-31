import cv2
import mediapipe as mp
import threading
import math
import time

from class_gameBar import GameBar





# Функция для вычисления расстояния между двумя точками на плоскости
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def draw_rounded_rectangle(img, rect, color, thickness=1, radius=10):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y + radius), (x + w, y + h - radius), color, thickness)
    cv2.rectangle(img, (x + radius, y), (x + w - radius, y + h), color, thickness)
    cv2.circle(img, (x + radius, y + radius), radius, color, thickness)
    cv2.circle(img, (x + w - radius, y + radius), radius, color, thickness)
    cv2.circle(img, (x + radius, y + h - radius), radius, color, thickness)
    cv2.circle(img, (x + w - radius, y + h - radius), radius, color, thickness)


# Импорт модулей MediaPipe для работы с руки и отслеживания жестов
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# переменные для отслеживания времени зажатия пальца
finger_pressed_time = None
finger_closed = False  # Переменная для отслеживания сжатия пальцев

# Начальные координаты и размеры прямоугольника на экране
rectangle_x = 30
rectangle_y = 30
rectangle_width = 180 #220
rectangle_height = 80 #100

# Частота кадров и статус активности игры
frame_rate = 30
game_active = False
fl = True

# Флаги для отслеживания состояния пальцев
pointer_finger_closed = False
thumb_finger_closed = False

# Определение переменных для дополнительных прямоугольников
green_rect = (30, 50, rectangle_width, rectangle_height)
blue_rect = (30, 150, rectangle_width, rectangle_height)
pink_rect = (30, 250, rectangle_width, rectangle_height)
red_rect = (30, 350, rectangle_width, rectangle_height)

yelow_rect = (400, 50, rectangle_width, rectangle_height)


try:
    while fl:
        if not game_active:
            # Инициализация видеозахвата с веб-камеры
            cap = cv2.VideoCapture(1)

            while cap.isOpened():
                ret, frame = cap.read()
                cv2.namedWindow("Menu", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Menu", 800, 600)
                frame = cv2.flip(frame, 1)  # Зеркальное отражение изображения

                if not ret:
                    break

                with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)

                    if results.multi_hand_landmarks:
                        for landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                            # Получаем координаты указательного и большого пальцев
                            finger8_x = int(landmarks.landmark[8].x * frame.shape[1])
                            finger8_y = int(landmarks.landmark[8].y * frame.shape[0])
                            finger4_x = int(landmarks.landmark[4].x * frame.shape[1])
                            finger4_y = int(landmarks.landmark[4].y * frame.shape[0])

                            # Вычисляем расстояние между пальцами
                            distance = calculate_distance((finger8_x, finger8_y), (finger4_x, finger4_y))

                            # Определение, открыты ли указательный и большой пальцы
                            # Определяем, закрыты ли указательный и большой пальцы
                            if distance < 30:
                                finger_open = True
                            else:
                                finger_open = False

                            if distance < 30:
                                finger_open = True
                            else:
                                finger_open = False

                            # Отслеживание событий с пальцами
                            if finger_open:
                                if not finger_closed:  # Если пальцы были открыты, а теперь сжаты
                                    finger_pressed_time = time.time()  # Запоминаем время начала сжатия
                                    finger_closed = True  # Обновляем статус сжатия пальцев

                                # Проверка времени сжатия
                                if time.time() - finger_pressed_time >= 0.8:
                                    # Запуск игры "Змейка"
                                    if rectangle_x < finger8_x < rectangle_x + rectangle_width and rectangle_y < finger8_y < rectangle_y + rectangle_height:
                                        GameBar.start_game_1()
                                        fl = False
                                        cv2.destroyWindow("Menu")
                                        # Сброс счётчика для следующей итерации
                                        finger_pressed_time = None
                                        finger_closed = False  # Пальцы разжались
                                        break

                                if time.time() - finger_pressed_time >= 1.5:
                                    # Запуск игры "Пин-Понг"
                                    if blue_rect[0] < finger8_x < blue_rect[0] + blue_rect[2] and blue_rect[1] < finger8_y < \
                                            blue_rect[1] + blue_rect[3]:
                                        cv2.putText(frame, "press 2", (blue_rect[0], blue_rect[1] - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                                        GameBar.start_game_2()
                                        fl = False
                                        cv2.destroyWindow("Menu")
                                        # Сброс счётчика для следующей итерации
                                        finger_pressed_time = None
                                        finger_closed = False  # Пальцы разжались
                                        break

                                # Запуск игры "Реакция"
                                if time.time() - finger_pressed_time >= 1.5:
                                    if pink_rect[0] < finger8_x < pink_rect[0] + pink_rect[2] and pink_rect[1] < finger8_y < \
                                            pink_rect[1] + pink_rect[3]:
                                        cv2.putText(frame, "press 3", (pink_rect[0], pink_rect[1] - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                                        GameBar.start_game_3()
                                        fl = False
                                        cv2.destroyWindow("Menu")
                                        # Сброс счётчика для следующей итерации
                                        finger_pressed_time = None
                                        finger_closed = False  # Пальцы разжались
                                        break

                                # Обработка действий внутри красного прямоугольника
                                if time.time() - finger_pressed_time >= 1.5:
                                    if red_rect[0] < finger8_x < red_rect[0] + red_rect[2] and red_rect[1] < finger8_y < \
                                            red_rect[1] + red_rect[3]:
                                        cv2.putText(frame, "press 4", (pink_rect[0], pink_rect[1] - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                                        GameBar.start_game_4()
                                        fl = False
                                        cv2.destroyWindow("Menu")
                                        # Сброс счётчика для следующей итерации
                                        finger_pressed_time = None
                                        finger_closed = False  # Пальцы разжались
                                        break

                                if time.time() - finger_pressed_time >= 1.5:
                                    if yelow_rect[0] < finger8_x < yelow_rect[0] + yelow_rect[2] and yelow_rect[
                                        1] < finger8_y < yelow_rect[1] + yelow_rect[3]:
                                        cv2.putText(frame, "press 5", (yelow_rect[0], yelow_rect[1] - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                                        GameBar.start_game_5()
                                        fl = False
                                        cv2.destroyWindow("Menu")
                                        # Сброс счётчика для следующей итерации
                                        finger_pressed_time = None
                                        finger_closed = False  # Пальцы разжались
                                        break
                            else:
                                # Если пальцы разжались до прошествия 2 секунд
                                finger_pressed_time = None
                                finger_closed = False  # Пальцы разжались

                overlay = frame.copy()


                ################################################################################################
                # Новые цвета для кнопок
                button_colors = [(255, 105, 180), (255, 208, 0), (89, 44, 212), (255, 0, 0), (0, 255, 0)]
                text_color = (255, 255, 255)  # Цвет текста на кнопках
                button_texts = ["РЕАКЦИЯ", "ДЕТЕКТ", "НАУЧНЫЙ", "ПИН-ПОНГ", "ЗМЕЙКА"]


                # Функция для создания закругленного прямоугольника
                # Отрисовка кнопок с новыми стилями
                for i, rect in enumerate([pink_rect, yelow_rect, red_rect, blue_rect, green_rect]):
                    button_color = button_colors[i]

                    # Создание закругленного прямоугольника кнопки
                    draw_rounded_rectangle(overlay, rect, button_color, thickness=-1, radius=20)

                    # Отрисовка текста на кнопке
                    text_size = cv2.getTextSize(button_texts[i], cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0]
                    text_x = rect[0] + int((rect[2] - text_size[0]) / 2)
                    text_y = rect[1] + 60
                    cv2.putText(overlay, button_texts[i], (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 1, text_color, 2)

                cv2.addWeighted(overlay, 0.5, frame, 1 - 0.5, 0, frame)

                cv2.imshow("Menu", frame)

                # Выход из приложения при нажатии клавиши "Esc"
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            cap.release()
except:
    print("Чет не то...")
