import os
import cv2
import pygame
import math
import numpy as np

def process_img(image, dim_x=128, dim_y=128):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    dim = (dim_x, dim_y)
    resized_img = cv2.resize(array, dim, interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    scaledImg = img_gray/255.
    mean, std = 0.5, 0.5
    normalizedImg = (scaledImg - mean) / std
    return normalizedImg

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def check_camera_switch():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return -1
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return -1
            elif event.key == pygame.K_1:
                return 1
            elif event.key == pygame.K_2:
                return 2
            elif event.key == pygame.K_3:
                return 3
            elif event.key == pygame.K_4:
                return 4
            elif event.key == pygame.K_5:
                return 5
            elif event.key == pygame.K_6:
                return 6
            elif event.key == pygame.K_7:
                return 7
            elif event.key == pygame.K_8:
                return 8
            elif event.key == pygame.K_9:
                return 9
    return 0

def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def correct_yaw(x):
    return(((x%360) + 360) % 360)

def create_folders(folder_names):
    for directory in folder_names:
        if not os.path.exists(directory):
                os.makedirs(directory)
