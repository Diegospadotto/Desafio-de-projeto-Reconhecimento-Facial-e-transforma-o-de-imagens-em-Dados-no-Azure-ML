from src.face_recognition import detect_faces
from src.data_loader import load_image
import os

# Caminho para a imagem de teste
image_path = os.path.join('data', 'test_image.jpg')

# Carregar a imagem
image = load_image(image_path)

# Detectar rostos na imagem
detect_faces(image)
