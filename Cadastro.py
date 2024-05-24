import face_recognition as fr
import cv2
import numpy as np
import os
import pickle

# Diretório onde as codificações faciais serão armazenadas
encodings_dir = "encodings"

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

# Capturar imagem da webcam para cadastrar um novo usuário
def register_user():
    video_capture = cv2.VideoCapture(0)
    print("Posicione o rosto na câmera e pressione 's' para capturar a imagem.")
    
    while True:
        ret, frame = video_capture.read()
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    if face_encodings:
        name = input("Digite o nome do usuário: ")
        known_face_encodings, known_face_names = load_encodings(f"{encodings_dir}/encodings.pkl")
        known_face_encodings.append(face_encodings[0])
        known_face_names.append(name)
        save_encodings(known_face_encodings, known_face_names, f"{encodings_dir}/encodings.pkl")
        print(f"Usuário {name} cadastrado com sucesso!")
    else:
        print("Nenhum rosto detectado. Tente novamente.")

# Chamar a função para registrar um novo usuário
register_user()
