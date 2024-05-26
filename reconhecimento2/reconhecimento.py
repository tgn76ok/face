import face_recognition
import cv2
import numpy as np
import os
import pickle

# Função para salvar novas faces e atualizar as listas de encodings e nomes
def save_new_face(face_encodings, name):
    if not os.path.exists('known_faces'):
        os.makedirs('known_faces')
    
    # Salvar todas as codificações de face em um único arquivo
    with open(f'known_faces/{name}.pkl', 'wb') as f:
        pickle.dump(face_encodings, f)
    
    known_face_encodings.extend(face_encodings)
    known_face_names.extend([name] * len(face_encodings))

# Função para carregar faces conhecidas do disco
def load_known_faces():
    if not os.path.exists('known_faces'):
        return [], []

    encodings = []
    names = []
    for file in os.listdir('known_faces'):
        if file.endswith('.pkl'):
            with open(os.path.join('known_faces', file), 'rb') as f:
                file_encodings = pickle.load(f)
                encodings.extend(file_encodings)
                names.extend([os.path.splitext(file)[0]] * len(file_encodings))
    return encodings, names

# Função para cadastrar novo usuário
def register_new_user(video_capture):
    print("Registrando novo usuário.")
    face_encodings = []
    while True:
        # Capturar um único quadro de vídeo
        ret, frame = video_capture.read()
        
        # Exibir o quadro
        cv2.imshow('Video', frame)
        
        # Pressionar 'c' para capturar a imagem do novo usuário
        if cv2.waitKey(1) & 0xFF == ord('c'):
            rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

            face_locations = face_recognition.face_locations(rgb_frame)
            if face_locations:
                encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if encodings:
                    face_encodings.append(encodings[0])
                    print(f"Imagem capturada. Total de imagens capturadas: {len(face_encodings)}")
                else:
                    print("Nenhum rosto encontrado. Tente novamente.")
            else:
                print("Nenhum rosto encontrado. Tente novamente.")
        
        # Pressionar 's' para salvar as imagens do novo usuário
        if cv2.waitKey(1) & 0xFF == ord('s'):
            if face_encodings:
                new_face_name = input("Digite o nome do novo usuário: ")
                save_new_face(face_encodings, new_face_name)
                print(f"Novo usuário {new_face_name} registrado com sucesso com {len(face_encodings)} imagens.")
                break
            else:
                print("Nenhuma imagem capturada. Tente novamente.")
    cv2.destroyAllWindows()

# Função para reconhecer rostos em tempo real
def recognize_faces(video_capture, known_face_encodings, known_face_names):
    process_this_frame = True
    while True:
        # Capture um único quadro de vídeo
        ret, frame = video_capture.read()

        # Redimensione o quadro de vídeo para 1/4 do tamanho para processamento de reconhecimento facial mais rápido
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Converta a imagem de cor BGR (usada pelo OpenCV) para cor RGB (usada pelo face_recognition)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Somente processe cada quadro alternado do vídeo para economizar tempo
        if process_this_frame:
            # Encontre todos os rostos e encodings de face no quadro atual do vídeo
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # Veja se a face é uma correspondência para o(s) rosto(s) conhecido(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Use a face conhecida com a menor distância para a nova face
                if True in matches:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Exiba os resultados
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Aumente novamente as localizações dos rostos, pois o quadro que detectamos foi redimensionado para 1/4 do tamanho
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Desenhe uma caixa ao redor do rosto
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Desenhe uma etiqueta com um nome abaixo do rosto
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Exiba a imagem resultante
        cv2.imshow('Video', frame)

        # Saia do loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Função de menu principal
def main_menu():
    print("Selecione uma opção:")
    print("1. Registrar novo usuário")
    print("2. Reconhecer faces em tempo real")
    print("3. Sair")
    
    choice = input("Digite o número da opção desejada: ")
    return choice

# Carregar faces conhecidas do disco
known_face_encodings, known_face_names = load_known_faces()

# Inicialize a webcam
video_capture = cv2.VideoCapture(0)

while True:
    choice = main_menu()
    if choice == '1':
        register_new_user(video_capture)
    elif choice == '2':
        recognize_faces(video_capture, known_face_encodings, known_face_names)
    elif choice == '3':
        break
    else:
        print("Opção inválida. Tente novamente.")

# Libere a captura de vídeo
video_capture.release()
cv2.destroyAllWindows()