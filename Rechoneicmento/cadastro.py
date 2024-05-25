import cv2
import os

def cadastrar_face(user_id, output_dir="Users"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    user_folder = os.path.join(output_dir, f'User.{user_id}')
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    sample_num = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Erro: Falha ao capturar imagem da câmera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Cadastrar Face", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(faces) > 0:
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                sample_num += 1
                cv2.imwrite(f"{user_folder}/User.{user_id}.{sample_num}.jpg", face)
                print(f"Imagem {sample_num} capturada")
        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# Exemplo de uso
user_id = input("Digite um ID numérico para a nova face: ")
cadastrar_face(user_id)