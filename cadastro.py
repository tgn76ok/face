import cv2
import dlib
import numpy as np
import os
import pickle

# Diretório onde as codificações faciais serão armazenadas
encodings_dir = "encodings"
os.makedirs(encodings_dir, exist_ok=True)

# Função para salvar codificações faciais no arquivo
def save_encodings(encodings, names, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump((encodings, names), f)

# Função para carregar codificações faciais do arquivo
def load_encodings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return [], []

# Inicializar o detector de rosto do dlib
detector = dlib.get_frontal_face_detector()
# Carregar o modelo de pontos de referência faciais do dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Carregar o modelo de reconhecimento facial do dlib
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Função para registrar um novo usuário
def register_user():
    video_capture = cv2.VideoCapture(0)
    print("Posicione o rosto na câmera e pressione 's' para capturar a imagem.")
    
    while True:
        ret, frame = video_capture.read()
        rgb_frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces2 = detector(rgb_frame2)
        
        # Desenhar retângulos em torno dos rostos detectados
        for face in faces2:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    
    if faces:
        shape = predictor(rgb_frame, faces[0])
        face_encoding = np.array(face_rec_model.compute_face_descriptor(rgb_frame, shape))
        
        name = input("Digite o nome do usuário: ")
        known_face_encodings, known_face_names = load_encodings(f"{encodings_dir}/encodings.pkl")
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
        save_encodings(known_face_encodings, known_face_names, f"{encodings_dir}/encodings.pkl")
        print(f"Usuário {name} cadastrado com sucesso!")
    else:
        print("Nenhum rosto detectado. Tente novamente.")


register_user()