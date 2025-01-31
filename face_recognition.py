from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import cv2
import matplotlib.pyplot as plt

# Configurações do Azure
face_api_key = "SUA_FACE_API_KEY"
endpoint = "SEU_ENDPOINT"

# Inicializar o cliente do Face API
face_client = FaceClient(endpoint, CognitiveServicesCredentials(face_api_key))

def detect_faces(image):
    """Detecta rostos em uma imagem usando o Azure Face API."""
    _, img_encoded = cv2.imencode('.jpg', image)
    stream = img_encoded.tobytes()

    detected_faces = face_client.face.detect_with_stream(
        stream, detection_model='detection_03'
    )

    if not detected_faces:
        print("Nenhum rosto detectado.")
        return

    # Exibe a imagem com os rostos detectados
    for face in detected_faces:
        rect = face.face_rectangle
        cv2.rectangle(image, (rect.left, rect.top), 
                      (rect.left + rect.width, rect.top + rect.height), 
                      (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
