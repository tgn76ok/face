import face_recognition as fr
import cv2
import numpy as np
import os

def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Carregar a imagem do arquivo
            image_path = os.path.join(known_faces_dir, filename)
            image = fr.load_image_file(image_path)

            # Obter as codificações do rosto
            face_encodings = fr.face_encodings(image)
            if face_encodings:  # Verifica se a lista não está vazia
                face_encoding = face_encodings[0]
                # Adicionar a codificação do rosto e o nome à lista
                known_face_encodings.append(face_encoding)
                known_face_names.append(os.path.splitext(filename)[0])

    return known_face_encodings, known_face_names

def reconhecer_faces(known_faces_dir):
    # Carregar e codificar os rostos conhecidos
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

    # Capturar vídeo da webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Converter o quadro de BGR para RGB
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        # Encontrar todas as localizações de rostos no quadro
        face_locations = fr.face_locations(rgb_frame)

        # Verifique se há localizações de rostos antes de tentar obter codificações
        face_encodings = []
        for face_location in face_locations:
            # Obter a codificação do rosto para cada localização detectada
            encodings = fr.face_encodings(rgb_frame, [face_location])
            if encodings:
                face_encodings.append(encodings[0])

        # Iterar sobre cada codificação de rosto encontrada no quadro
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = fr.compare_faces(known_face_encodings, face_encoding)
            name = "Desconhecido"

            # Se um rosto conhecido for encontrado
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Desenhar uma caixa ao redor do rosto
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Desenhar uma etiqueta com o nome abaixo do rosto
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Exibir a imagem resultante
        cv2.imshow('Video', frame)

        # Pressione 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar a captura de vídeo e fechar as janelas
    video_capture.release()
    cv2.destroyAllWindows()

# Exemplo de uso
known_faces_dir = "/home/tigaz/Documentos/myprojetos/face/reconhimento_rosto/Users/tigaz.4"
reconhecer_faces(known_faces_dir)
