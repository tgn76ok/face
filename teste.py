import face_recognition as fr
import cv2
import numpy
import os

def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = fr.load_image_file(os.path.join(known_faces_dir, filename))
            face_encodings = fr.face_encodings(image)
            if face_encodings:  # Verifica se a lista não está vazia
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])  # Usa o nome do arquivo como nome da pessoa
            else:
                print(f"Nenhum rosto encontrado na imagem {filename}")
    return known_face_encodings, known_face_names
# Diretório onde as fotos conhecidas estão armazenadas
known_faces_dir = "/home/tigaz/Documentos/myprojetos/face/reconhimento_rosto/MyFace"

# Carregar e codificar os rostos conhecidos
known_face_encodings, known_face_names = load_known_faces(known_faces_dir)



# Capturar vídeo da webcam
video_capture = cv2.VideoCapture(0)




while True:
    # Capturar um único quadro de vídeo
    ret, frame = video_capture.read()

    # Converter a imagem de BGR (usado pelo OpenCV) para RGB (usado pelo face_recognition)
    rgb_frame = numpy.ascontiguousarray(frame[:, :, ::-1])

    # Encontrar todas as localizações de rostos no quadro atual do vídeo
    face_locations = fr.face_locations(rgb_frame)

    # Encontrar os marcos faciais (landmarks) para cada rosto no quadro atual do vídeo
    face_landmarks_list = fr.face_landmarks(rgb_frame, face_locations)

    
    
    # Encontrar as codificações dos rostos para cada rosto no quadro atual do vídeo
    face_encodings = [fr.face_encodings(rgb_frame, [face_location])[0] for face_location in face_locations]

    # Exibir os resultados
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Desenhar um retângulo ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Adicionar um texto indicando que um rosto foi reconhecido
        cv2.putText(frame, "Rosto reconhecido", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Exibir a imagem resultante
    cv2.imshow('Video', frame)

    # Pressionar 'q' no teclado para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar o controle da webcam
video_capture.release()
cv2.destroyAllWindows()
