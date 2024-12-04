import os
import sys
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_knn():
    knn_module = importlib.import_module("modelos.knn_model")
    knn_module.run()

def run_decision_tree():
    dt_module = importlib.import_module("modelos.decision_tree_model")
    dt_module.run()

def run_random_forest():
    rf_module = importlib.import_module("modelos.random_forest_model")
    rf_module.run()

def main():
    while True:
        print("\nEscolha uma opção:")
        print("1 - Rodar o modelo k-NN")
        print("2 - Rodar o modelo Decision Tree")
        print("3 - Rodar o modelo Random Forest")
        print("4 - Rodar todos os modelos")
        print("0 - Sair")
        
        choice = input("Digite sua escolha: ")
        
        if choice == '1':
            run_knn()
        elif choice == '2':
            run_decision_tree()
        elif choice == '3':
            run_random_forest()
        elif choice == '4':
            print("\nExecutando todos os modelos...")
            run_knn()
            run_decision_tree()
            run_random_forest()
        elif choice == '0':
            print("\nSaindo...")
            break
        else:
            print("\nOpção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
