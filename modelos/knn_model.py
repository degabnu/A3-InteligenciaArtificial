import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

def run():
    sns.set(style="whitegrid")

    output_dir = './imagens/knn'
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

    plt.figure(figsize=(12, 6))
    sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='blue', label='Normal', kde=True, stat="density")
    sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='red', label='Fraude', kde=True, stat="density")
    plt.title('Distribuição de Montantes para Transações Normais e Fraudulentas')
    plt.xlabel('Montante')
    plt.ylabel('Densidade')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'distribuicao_montante.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.histplot(df[df['Class'] == 0]['Time'], bins=50, color='blue', label='Normal', kde=True, stat="density")
    sns.histplot(df[df['Class'] == 1]['Time'], bins=50, color='red', label='Fraude', kde=True, stat="density")
    plt.title('Distribuição de Tempo para Transações Normais e Fraudulentas')
    plt.xlabel('Tempo')
    plt.ylabel('Densidade')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'distribuicao_tempo.png'))
    plt.close()

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

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)

    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))
    print("\nAcurácia do modelo:")
    print(accuracy_score(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nMatriz de Confusão:")
    print(conf_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fraude'], yticklabels=['Normal', 'Fraude'])
    plt.title('Matriz de Confusão - k-NN')
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Real')
    plt.savefig(os.path.join(output_dir, 'matriz_confusao_knn.png'))
    plt.close()

    print("\nGráficos salvos na pasta ./imagens/knn")
