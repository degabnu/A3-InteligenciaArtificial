import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

sns.set(style="whitegrid")

def run():
    output_dir = './imagens/decision_tree'
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv("./dados/creditcard.csv")

    print("Primeiras linhas do dataset:")
    print(df.head())

    print("\nInformações do dataset:")
    print(df.info())

    print("\nEstatísticas descritivas:")
    print(df.describe())

    print("\nDistribuição das classes (0 = normal, 1 = fraude):")
    print(df['Class'].value_counts(normalize=True))

    plot_data_distributions(df, output_dir)
    plot_pca_projection(df, output_dir)

    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    y_pred = dt_model.predict(X_test)

    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))
    print("\nAcurácia do modelo:")
    print(accuracy_score(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nMatriz de Confusão:")
    print(conf_matrix)
    plot_confusion_matrix(conf_matrix, output_dir)

    print("\nGráficos salvos na pasta ./imagens/decision_tree")

def plot_data_distributions(df, output_dir):
    plt.figure(figsize=(12, 6))
    sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='blue', label='Normal', kde=True, stat="density")
    sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='red', label='Fraude', kde=True, stat="density")
    plt.title('Distribuição de Montantes para Transações Normais e Fraudulentas')
    plt.xlabel('Montante')
    plt.ylabel('Densidade')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'distribuicao_montante.png'))
    plt.close()

def plot_pca_projection(df, output_dir):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop('Class', axis=1))
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Class', data=df, palette='Set1', alpha=0.6)
    plt.title('PCA - Projeção das Transações em 2D')
    plt.savefig(os.path.join(output_dir, 'pca_projecao.png'))
    plt.close()

def plot_confusion_matrix(conf_matrix, output_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fraude'], yticklabels=['Normal', 'Fraude'])
    plt.title('Matriz de Confusão - Decision Tree')
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Real')
    plt.savefig(os.path.join(output_dir, 'matriz_confusao_decision_tree.png'))
    plt.close()

if __name__ == "__main__":
    run()
