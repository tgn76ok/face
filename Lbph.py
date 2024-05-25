from PIL import Image
import cv2
import numpy as np
import os

# Função para obter dados de imagem
def get_image_data():
    paths = []
    for f in os.listdir('Users/'):
        for n in os.listdir(f"Users/{f}/"):
            paths.append(f'Users/{f}/{n}')
            
    faces = []
    ids = []
    
    for path in paths:
        image = Image.open(path).convert('L')  # Converter para escala de cinza
        image_np = np.array(image, 'uint8')
        id = int(path.split('.')[1])

        ids.append(id)
        faces.append(image_np)

    return np.array(ids), faces

# Obter os dados de imagem e IDs
ids, faces = get_image_data()

# Treinar o classificador LBPH
lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.train(faces, ids)
lbph_classifier.write('lbph_classifier.yml')

# Carregar o classificador treinado
lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read('lbph_classifier.yml')

# Caminhos para as imagens de teste
paths = []
for f in os.listdir('Users/'):
    for n in os.listdir(f"Users/{f}/"):
        paths.append(f'Users/{f}/{n}')

# Prever e mostrar os resultados
for path in paths:
    image = Image.open(path).convert('L')  # Converter para escala de cinza
    image_np = np.array(image, 'uint8')
    prediction, _ = lbph_face_classifier.predict(image_np)
    expected_output = int(path.split('.')[1])

    cv2.putText(image_np, 'Pred: ' + str(prediction), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
    cv2.putText(image_np, 'Exp: ' + str(expected_output), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

    
    cv2.imshow('Resultado', image_np)  # Nome da janela corrigido
    cv2.waitKey(2000)  # Espera por 2 segundos antes de mostrar a próxima imagem

cv2.destroyAllWindows()