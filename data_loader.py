import cv2

def load_image(image_path):
    """Carrega uma imagem usando OpenCV."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar imagem: {image_path}")
    return image
