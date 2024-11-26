import torch
from ultralytics import YOLO
import cv2
import numpy as np

model_path = 'runs/segment/train/weights/best.pt'
model = YOLO(model_path)


image_path = 'prueba.jpg'
image = cv2.imread(image_path)
if image is None:
    print(f"Error al cargar la imagen desde la ruta: {image_path}")
    exit()


results = model(image)

# USAR la máscar xy q ya se encentra en pixeles
for r in results:
    masks = r.masks  
    if masks is not None:
        # Iterar sobre cada máscara
        for mask in masks.xy:  
            contour = np.array(mask, dtype=np.int32)
            cv2.polylines(image, [contour], isClosed=True, color=(0, 255, 0), thickness=2)


cv2.imshow('Contornos', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

output_path = 'imagen_con_contornos.jpg'
cv2.imwrite(output_path, image)
print(f"Imagen con contornos guardada en: {output_path}")
