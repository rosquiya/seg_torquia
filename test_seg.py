import torch
from ultralytics import YOLO
import cv2
import numpy as np

#model_path = 'runs/detect/train/weights/best.pt'
model_path = 'runs/segment/train/weights/best.pt'
model = YOLO(model_path)


image_path = 'prueba.jpg'


image = cv2.imread(image_path)
if image is None:
    print(f"Error al cargar la imagen desde la ruta: {image_path}")
    exit()


results = model(image)


annotated_image = results[0].plot()  


output_path = 'prediccion_seg.jpg'
cv2.imwrite(output_path, annotated_image)
print(f"Imagen guardada en: {output_path}")


extreme_points = []

# Iterar sobre cada una de las segmentaciones
for i, result in enumerate(results.xywh[0]):
    # Obtener la máscara de la segmentación
    mask = result.masks  # Obtén la máscara de la segmentación
    
    if mask is not None:
        # Convierte la máscara a una imagen binaria
        mask_binary = mask[0].cpu().numpy().astype(np.uint8)

        # Encuentra los contornos de la máscara
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Obtener los puntos más extremos de la segmentación
            x_min = np.min(contour[:, 0, 0])  # Punto más a la izquierda
            x_max = np.max(contour[:, 0, 0])  # Punto más a la derecha
            y_min = np.min(contour[:, 0, 1])  # Punto más arriba
            y_max = np.max(contour[:, 0, 1])  # Punto más abajo

            # Guardar los puntos extremos
            extreme_points.append((x_min, y_min, x_max, y_max))

            # Mostrar los puntos extremos (opcional)
            print(f"Segmentación {i}:")
            print(f"  Puntos extremos: ({x_min}, {y_min}), ({x_max}, {y_max})")