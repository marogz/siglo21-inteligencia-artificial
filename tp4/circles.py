import cv2
import numpy as np
import matplotlib.pyplot as plt


INPUT_PATH = 'engine.jpg'
ACCUMULATOR_THRESHOLD = 100 # Use 60 for example_circles.jpg image


def hough_transform_circles(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar un desenfoque para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar la detección de bordes con el operador de Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Aplicar la transformada de Hough para detectar círculos
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=150,
        param2=ACCUMULATOR_THRESHOLD,
        minRadius=0,
        maxRadius=0
    )

    # Dibujar los círculos detectados sobre la imagen original
    result = image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(result, center, radius, (0, 255, 0), 2)
            cv2.circle(result, center, 2, (0, 0, 255), 3)

    return result


image = cv2.imread(INPUT_PATH)

# Transformada de Hough para circunferencias
result_image = hough_transform_circles(image)

plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)), plt.title('Hough Transform Circles')
plt.show()
