from cadastro import register_user
from reconhecimento import recognize_faces

def main():
    print("Escolha uma opção: ")
    print("1: Registrar novo usuário")
    print("2: Reconhecer rostos")

    opcao = input("Digite o número da opção desejada: ")

    if opcao == '1':
        register_user()
    elif opcao == '2':
        recognize_faces()
    else:
        print("Opção inválida. Saindo do programa.")

if __name__ == "__main__":
    main()