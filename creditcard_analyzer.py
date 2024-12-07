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

def run_gradient_boosting():
    gb_module = importlib.import_module("modelos.gradient_boosting")
    gb_module.run()

def run_svm():
    svm_module = importlib.import_module("modelos.svm")
    svm_module.run()

def run_regression():
    regression_module = importlib.import_module("modelos.regression")
    regression_module.run()

def main():
    while True:
        print("\nEscolha uma opção:")
        print("1 - Rodar o modelo k-NN")
        print("2 - Rodar o modelo Decision Tree")
        print("3 - Rodar o modelo Random Forest")
        print("4 - Rodar o modelo Regressão Logística")
        print("5 - Rodar o modelo SVM")
        print("6 - Rodar o modelo Gradient Boosting")
        print("7 - Rodar todos os modelos")
        print("0 - Sair")
        
        choice = input("Digite sua escolha: ")
        
        if choice == '1':
            run_knn()
        elif choice == '2':
            run_decision_tree()
        elif choice == '3':
            run_random_forest()
        elif choice == '4':
            run_regression()
        elif choice == '5':
            run_svm()
        elif choice == '6':
            run_gradient_boosting()
        elif choice == '7':
            print("\nExecutando todos os modelos...")
            run_knn()
            run_decision_tree()
            run_random_forest()
            run_gradient_boosting()
            run_svm()
            run_regression()
        elif choice == '0':
            print("\nSaindo...")
            break
        else:
            print("\nOpção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
