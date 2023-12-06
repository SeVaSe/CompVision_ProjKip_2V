import subprocess
import sys

import cv2
import mediapipe as mp
import numpy as np


def run_detected():
    try:
        mp_pose = mp.solutions.pose

        # Инициализация модели
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Создание цветного фона
            color_image = np.zeros_like(image)

            # Обработка изображения
            results = pose.process(image)

            if results.pose_landmarks is not None:
                # Рисование линий позы на цветном фоне
                mp.solutions.drawing_utils.draw_landmarks(
                    color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Вывод изображения с позой и отдельно скилета
                cv2.imshow('Pose Detection', image)
                cv2.imshow('Skeleton in Color', color_image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        # Освобождение ресурсов
        cap.release()
        cv2.destroyAllWindows()

        # Освобождение ресурсов Mediapipe
        pose.close()
        subprocess.run(["C:/PYTHON_/_PROJECT_PYTHON/Python_Project_Other/CompVision_ProjKip_2V/venv/Scripts/python.exe",
                        "menu.py"])
        sys.exit()

    except:
        cap.release()
        cv2.destroyAllWindows()
        subprocess.run(["C:/PYTHON_/_PROJECT_PYTHON/Python_Project_Other/CompVision_ProjKip_2V/venv/Scripts/python.exe",
                        "menu.py"])
        sys.exit()


