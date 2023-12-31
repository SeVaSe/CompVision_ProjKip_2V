import cv2

from game_snake import run_snake_game
from game_pin_pong import run_pin_pong_game
from game_circle_reaction import run_circle_reaction_game
from science_project_growth import run_science_project
from science_detected_peop import run_detected


class GameBar:
    """КЛАСС ДЛЯ ЗАПУСКА ПРОЕКТОВ"""
    @staticmethod
    def start_game_1():
        """Метод для запуска Змейки"""
        global game_active
        game_active = True
        cv2.destroyWindow("Menu")
        run_snake_game()

    @staticmethod
    def start_game_2():
        """Метод для запуска Пин-Понг"""
        global game_active
        game_active = True
        cv2.destroyWindow("Menu")
        run_pin_pong_game()

    @staticmethod
    def start_game_3():
        """Метод для запуска Реакции"""
        global game_active
        game_active = True
        cv2.destroyWindow("Menu")
        run_circle_reaction_game()

    @staticmethod
    def start_game_4():
        """Метод для запуска Научного проекта"""
        global game_active
        game_active = True
        cv2.destroyWindow("Menu")
        run_science_project()

    @staticmethod
    def start_game_5():
        """Метод для запуска Научного проекта"""
        global game_active
        game_active = True
        cv2.destroyWindow("Menu")
        run_detected()

    @staticmethod
    def start_game_6():
        """Метод для запуска Научного проекта"""
        global game_active
        game_active = True
        cv2.destroyWindow("Menu")
        run_detected()

