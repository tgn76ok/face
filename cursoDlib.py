from PIL import Image
import cv2
import numpy as np
import os
import dlib

# Inicializar o detector de rosto do dlib
detector_face = dlib.get_frontal_face_detector()
# Carregar o modelo de pontos de referência faciais do dlib
detector_pontos = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Carregar o modelo de reconhecimento facial do dlib
descritor_facial_extrator = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

confianca = 0.5
previsoes = []
saidas_esperadas = []

index = {}
idx = 0
descritores_faciais = None

# Carregar e processar imagens para criar descritores faciais
paths = [os.path.join('/home/tigaz/Documentos/myprojetos/face/reconhimento_rosto/yalefaces/yalefaces/train', f) for f in os.listdir('/home/tigaz/Documentos/myprojetos/face/reconhimento_rosto/yalefaces/yalefaces/train')]
for path in paths:
    imagem = Image.open(path).convert('RGB')
    imagem_np = np.array(imagem, 'uint8')
    deteccoes = detector_face(imagem_np, 1)
    for face in deteccoes:
        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(imagem_np, (l, t), (r, b), (0, 0, 255), 2)

        pontos = detector_pontos(imagem_np, face)
        for ponto in pontos.parts():
            cv2.circle(imagem_np, (ponto.x, ponto.y), 2, (0, 255, 0), 1)

        descritor_facial = descritor_facial_extrator.compute_face_descriptor(imagem_np, pontos)
        descritor_facial = np.asarray(descritor_facial, dtype=np.float64)
        descritor_facial = descritor_facial[np.newaxis, :]

        if descritores_faciais is None:
            descritores_faciais = descritor_facial
        else:
            descritores_faciais = np.concatenate((descritores_faciais, descritor_facial), axis=0)

        index[idx] = path
        idx += 1
    # cv2.imshow('Imagem', imagem_np)  # Para visualização intermediária

# Carregar e processar imagens para reconhecimento
paths2 = [os.path.join('/home/tigaz/Documentos/myprojetos/face/reconhimento_rosto/yalefaces/yalefaces/test', f) for f in os.listdir('/home/tigaz/Documentos/myprojetos/face/reconhimento_rosto/yalefaces/yalefaces/test')]
for path in paths2:
    imagem = Image.open(path).convert('RGB')
    imagem_np = np.array(imagem, 'uint8')
    deteccoes = detector_face(imagem_np, 1)
    for face in deteccoes:
        pontos = detector_pontos(imagem_np, face)
        descritor_facial = descritor_facial_extrator.compute_face_descriptor(imagem_np, pontos)
        descritor_facial = np.asarray(descritor_facial, dtype=np.float64)
        descritor_facial = descritor_facial[np.newaxis, :]

        distancias = np.linalg.norm(descritor_facial - descritores_faciais, axis=1)
        indice_minimo = np.argmin(distancias)
        distancia_minima = distancias[indice_minimo]
        if distancia_minima <= confianca:
            nome_previsao = int(os.path.split(index[indice_minimo])[1].split('.')[0].replace('subject', ''))
        else:
            nome_previsao = 'Face não identificada'

        nome_real = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))

        previsoes.append(nome_previsao)
        saidas_esperadas.append(nome_real)

        cv2.putText(imagem_np, 'Pred: ' + str(nome_previsao), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
        cv2.putText(imagem_np, 'Exp: ' + str(nome_real), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

        cv2.imshow('Resultado', imagem_np)  # Ordem correta dos argumentos

previsoes = np.array(previsoes)
saidas_esperadas = np.array(saidas_esperadas)
print(previsoes,'=',saidas_esperadas)
# Esperar pressionar uma tecla para fechar as janelas
cv2.waitKey(0)
cv2.destroyAllWindows()
