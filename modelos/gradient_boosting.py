import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

sns.set(style="whitegrid")

def run():
    output_dir = './imagens/gradient_boosting'
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv("./dados/creditcard.csv")

    print("Primeiras linhas do dataset:")
    print(df.head())

    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)

    y_pred = gb_model.predict(X_test)

    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))
    print("\nAcurácia do modelo:")
    print(accuracy_score(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nMatriz de Confusão:")
    print(conf_matrix)
    plot_confusion_matrix(conf_matrix, output_dir)

    print("\nGráficos salvos na pasta ./imagens/gradient_boosting")

def plot_confusion_matrix(conf_matrix, output_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fraude'], yticklabels=['Normal', 'Fraude'])
    plt.title('Matriz de Confusão')
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Real')
    plt.savefig(os.path.join(output_dir, 'matriz_confusao.png'))
    plt.close()

if __name__ == "__main__":
    run()
