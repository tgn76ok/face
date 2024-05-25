import cv2
import face_recognition
import numpy as np

# Carregar imagem de exemplo com faces conhecidas
known_image = face_recognition.load_image_file("/home/tigaz/Documentos/myprojetos/face/reconhimento_rosto/Users/User.4/User.4.16.jpg")
known_face_encodings = face_recognition.face_encodings(known_image)
known_face_names = ["Person 1", "Person 2", "Person 3"]  # Nomes correspondentes

# Iniciar a captura de vídeo
video_capture = cv2.VideoCapture(0)

while True:
    # Capturar frame do vídeo
    ret, frame = video_capture.read()

    # Converter a imagem de BGR (OpenCV) para RGB (face_recognition)
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

    # Encontrar todas as faces e codificações de face no frame atual de vídeo
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop através de cada face encontrada no frame atual de vídeo
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Verificar se a face corresponde a alguma face conhecida
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Se uma correspondência foi encontrada na known_face_encodings
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Desenhar um retângulo ao redor da face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Desenhar uma etiqueta com o nome abaixo da face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Mostrar o resultado
    cv2.imshow('Video', frame)

    # Pressionar 'q' no teclado para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo
video_capture.release()
cv2.destroyAllWindows()
