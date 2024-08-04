
COLORS_NAMES = {
    0: 'threshold',
    1: 'orange',
    2: 'blue',
    3: 'yellow',
    4: 'red',
    5: 'green',
    6: 'pink',
    7: 'cyan'
}

COLORS = [
    [-1, -1, -1],  # threshold
    [0, 128, 255],  # orange 
    [255, 0, 0],   # blue
    [0, 255, 255], # yellow
    [0, 0, 255],   # red
    [0, 255, 0],   # green
    [255, 0, 255], # pink
    [255, 255, 0], # cyan
]

IMG_SIZE = (640, 480)

TEMPLATES_FOLDER = 'images/templates/all/'

def read_folder(folder=TEMPLATES_FOLDER):
    import os
    import cv2

    images = []
    answers = []
    images_names = []
    ext = '.png'
    for filename in os.listdir(folder):
        img_name = filename[:-len(ext)]
        if filename.endswith(ext):
            ans_path = folder + '/answers/ans_' + filename

            if not os.path.exists(ans_path):
                print(f"Answer not found for {filename}")
                continue

            img = cv2.imread(folder + '/' + filename)
            images.append(cv2.resize(img, IMG_SIZE))

            ans = cv2.imread(ans_path)
            answers.append(cv2.resize(ans, IMG_SIZE))

            images_names.append(img_name)

    return images, answers, images_names

def filter_color(img, color):
    import numpy as np

    mask = (img[:, :, 0] == color[0]) & (img[:, :, 1] == color[1]) & (img[:, :, 2] == color[2])
    filtered_img = np.zeros_like(img)
    filtered_img[mask] = color

    return filtered_img

def filter_all_colors(img):
    filtered = []
    for i, color in enumerate(COLORS):
        filtered.append(filter_color(img, color))
    return filtered
