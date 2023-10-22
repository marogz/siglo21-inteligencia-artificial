import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

MODEL_PATH = 'model.jpg'
INPUT_PATH = 'input.jpg'
SCALE_PERCENT = 100
THRESHOLD = 127

def train_hopfield(image):
    num_pixels = image.shape[0] * image.shape[1]

    # Crear una matriz de pesos sinápticos para el modelo de Hopfield.
    weight_matrix = np.zeros((num_pixels, num_pixels))

    # Binarizar la imagen de entrenamiento.
    _, binary_image = cv2.threshold(image, THRESHOLD, 1, cv2.THRESH_BINARY)
    binary_flat = binary_image.flatten()
    
    # Entrenamiento del modelo de Hopfield (Aprendizaje de Hebbian).
    for i in tqdm(range(num_pixels), 'Entrenando'):
        for j in range(num_pixels):
            if i != j:
                # Fortalecer la conexión si los píxeles tienen el mismo valor (Aprendizaje Hebbiano).
                weight_matrix[i, j] += binary_flat[i] * binary_flat[j]

    return weight_matrix

def update_hopfield(image, weight_matrix):
    # Binarizar la imagen de entrada.
    _, binary_image = cv2.threshold(image, THRESHOLD, 1, cv2.THRESH_BINARY)

    binary_flat = binary_image.flatten()

    hopfield_state = binary_flat.copy()

    max_iterations = 100
    for _ in range(max_iterations):
        updated_state = hopfield_state.copy()

        # Actualización iterativa de las neuronas del modelo de Hopfield.
        for i in range(binary_flat.shape[0]):
            activation = np.dot(weight_matrix[i], hopfield_state)
            updated_state[i] = np.array(1 if activation > 0 else -1, dtype=np.int8)

        # Comprobar si el modelo ha alcanzado un estado estable.
        if np.array_equal(updated_state, hopfield_state):
            break
        else:
            hopfield_state = updated_state

    return hopfield_state


def find_ring_center(image):
    _, binary_image = cv2.threshold(image, THRESHOLD, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])
            return center_x, center_y
    return None

image_model = cv2.imread(MODEL_PATH, cv2.IMREAD_GRAYSCALE)

width = int(image_model.shape[1] * SCALE_PERCENT / 100)
height = int(image_model.shape[0] * SCALE_PERCENT / 100)
dim = (width, height)

image_model = cv2.resize(image_model, dim)
weight_matrix = train_hopfield(image_model)

image_input = cv2.imread(INPUT_PATH, cv2.IMREAD_GRAYSCALE)
image_input = cv2.resize(image_input, (image_model.shape[1], image_model.shape[0]))

hopfield_state = update_hopfield(image_input, weight_matrix)

state_image = (hopfield_state.reshape((image_input.shape[0], image_input.shape[1])) * 255).astype(np.uint8)

center = find_ring_center(state_image)

plt.imshow(state_image, cmap='gray')
plt.title('Hopfield State')
if center:
    plt.plot(center[0], center[1], 'ro')
plt.show()
