import cv2
import dlib
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

# Inicializar o detector de rosto do dlib
detector = dlib.get_frontal_face_detector()
# Carregar o modelo de pontos de referência faciais do dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Carregar o modelo de reconhecimento facial do dlib
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Função para reconhecer rostos em tempo real
def recognize_faces():
    known_face_encodings, known_face_names = load_encodings(f"{encodings_dir}/encodings.pkl")

    # Capturar vídeo da webcam
    video_capture = cv2.VideoCapture(0)

    # Configurações para melhorar a precisão e performance
    tolerance = 0.5
    frame_resizing_scale = 0.25  # Reduz o tamanho do frame para acelerar o processamento

    while True:
        # Capturar um único quadro de vídeo
        ret, frame = video_capture.read()
        rgb_frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces2 = detector(rgb_frame2)
        
        # Desenhar retângulos em torno dos rostos detectados
        for face in faces2:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Video', frame)
        
        # Redimensionar o quadro de vídeo para 1/4 do tamanho para processamento mais rápido
        small_frame = cv2.resize(frame, (0, 0), fx=frame_resizing_scale, fy=frame_resizing_scale)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Encontrar todas as localizações de rostos no quadro redimensionado
        faces = detector(rgb_small_frame)
        
        face_encodings = []
        face_locations = []
        
        for face in faces:
            print('------------')
            shape = predictor(rgb_small_frame, face)
            face_encoding = np.array(face_rec_model.compute_face_descriptor(rgb_small_frame, shape))
            face_encodings.append(face_encoding)
            face_locations.append((face.top(), face.right(), face.bottom(), face.left()))
        
        # Ajustar as coordenadas dos rostos para o tamanho original do quadro
        face_locations = [(int(top / frame_resizing_scale), int(right / frame_resizing_scale), int(bottom / frame_resizing_scale), int(left / frame_resizing_scale)) for (top, right, bottom, left) in face_locations]

        # Exibir os resultados
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Verificar se o rosto encontrado corresponde a um rosto conhecido
            distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
            matches = distances <= tolerance
            name = "Desconhecido"

            # Se houver correspondência, usar o primeiro rosto correspondente encontrado
            if any(matches):
                best_match_index = np.argmin(distances)
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


recognize_faces()