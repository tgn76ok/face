import cv2
import numpy as np
import os
from PIL import Image

# Função para obter dados de imagem
def get_image_data():
    paths = []
    for f in os.listdir('Users/'):
        for n in os.listdir(f"Users/{f}/"):
            user_folder = os.path.join("Users", f)
            if os.path.isdir(user_folder):
                for filename in os.listdir(user_folder):
                    paths.append(os.path.join(user_folder, filename))
            
    faces = []
    ids = []
    
    for path in paths:
        image = Image.open(path).convert('L')  # Converter para escala de cinza
        image_np = np.array(image, 'uint8')
        id = int(os.path.basename(path).split('.')[1])  # Extrair o ID do nome do arquivo

        ids.append(id)
        faces.append(image_np)

    return np.array(ids), faces

# Função para cadastrar faces
def cadastrar_faces():
    if not os.path.exists('Users'):
            os.makedirs('Users')

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return
    
    id = input("Digite um ID numérico para a nova face: ")
    nome = input("Digite o seu nome: ")
    user_folder = os.path.join('Users', f'{nome}.{id}')
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

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
                cv2.imwrite(f"{user_folder}/User.{id}.{sample_num}.jpg", face)
                print(f"Imagem {sample_num} capturada")
        elif key == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
# Função para treinar o classificador LBPH
def treinar_classificador():
    ids, faces = get_image_data()
    lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
    lbph_classifier.train(faces, ids)
    lbph_classifier.write('lbph_classifier.yml')
    print("Classificador treinado e salvo com sucesso.")

# Função para reconhecer faces em tempo real
def reconhecer_faces():
    lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
    lbph_face_classifier.read('lbph_classifier.yml')

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Erro: Falha ao capturar imagem da câmera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            prediction, conf = lbph_face_classifier.predict(roi_gray)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {prediction}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {prediction}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        cv2.imshow("Reconhecer Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()


# Menu principal
def main():
    while True:
        print("Selecione uma opção:")
        print("1. Cadastrar nova face")
        print("2. Treinar classificador")
        print("3. Reconhecer face em tempo real")
        print("4. Sair")
        
        choice = input("Digite sua escolha: ")
        
        if choice == '1':
            cadastrar_faces()
        elif choice == '2':
            treinar_classificador()
        elif choice == '3':
            reconhecer_faces()
        elif choice == '4':
            break
        else:
            print("Escolha inválida. Tente novamente.")

if __name__ == "__main__":
    main()
