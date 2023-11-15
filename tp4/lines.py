import cv2
import numpy as np
import matplotlib.pyplot as plt


INPUT_PATH = 'engine.jpg'
ACCUMULATOR_THRESHOLD = 120 # Use 200 for example_lines.jpg image


def hough_transform_lines(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar un desenfoque para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar la detección de bordes con el operador de Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Aplicar la transformada de Hough para detectar líneas
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=ACCUMULATOR_THRESHOLD)

    # Dibujar las líneas detectadas sobre la imagen original
    result = image.copy()
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return result


image = cv2.imread(INPUT_PATH)

# Transformada de Hough para rectas
result_image = hough_transform_lines(image)

plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)), plt.title('Hough Transform Lines')
plt.show()
