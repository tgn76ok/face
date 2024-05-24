import face_recognition as fr
import cv2
import numpy as np
import os
import pickle

# Diretório onde as codificações faciais estão armazenadas
encodings_dir = "encodings"

# Função para carregar codificações faciais do arquivo
def load_encodings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return [], []

# Carregar codificações faciais conhecidas
known_face_encodings, known_face_names = load_encodings(f"{encodings_dir}/encodings.pkl")

# Capturar vídeo da webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capturar um único quadro de vídeo
    ret, frame = video_capture.read()

    # Converter a imagem de BGR (usado pelo OpenCV) para RGB (usado pelo face_recognition)
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

    # Encontrar todas as localizações de rostos no quadro atual do vídeo
    face_locations = fr.face_locations(rgb_frame)

    # Encontrar as codificações dos rostos para cada rosto no quadro atual do vídeo
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    # Exibir os resultados
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Verificar se o rosto encontrado corresponde a um rosto conhecido
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        name = "Desconhecido"

        # Se houver correspondência, usar o primeiro rosto correspondente encontrado
        face_distances = fr.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Desenhar um retângulo ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Adicionar um texto indicando o nome do rosto reconhecido
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Exibir a imagem resultante
    cv2.imshow('Video', frame)

    # Pressionar 'q' no teclado para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar o controle da webcam
video_capture.release()
cv2.destroyAllWindows()