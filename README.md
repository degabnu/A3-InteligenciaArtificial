# Inteligência Artificial 
- Aluno: Douglas Cristiano da Silva 
- RA: 152211260

# Projeto Final A3: Desenvolvimento de um Agente Inteligente

## Instalação de libs
pip install seaborn pandas matplotlib

## Execução do projeto
python .\creditcard_analyzer.py

## Definição do Problema
O objetivo deste projeto é identificar transações fraudulentas no conjunto de dados "Credit Card Fraud Detection", disponível [aqui](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download&select=creditcard.csv), utilizando o modelo de classificação Random Forest.

## Estrutura do projeto

src
    dados - contem o dataset utilizado no projeto.
    imagens - contem imagens geradas pelos modelos.
        decision_tree
        knn
        random_forest
    modelos - contem os scripts dos modelos utilizados para as analises
        random_forest.py
        decision_tree.py
        knn.py
creditcard_analyzer.py - script principal do projeto
README.md

## Análise dos Dados
- O dataset se encontra na pasta "dados" e contém 284807 transações, das quais apenas 0.17% são fraudulentas.
- As features V1 a V28 foram obtidas a partir de uma transformação PCA das features originais, devido a questões de privacidade e segurança.
- A variável 'Amount' representa o valor da transação, e 'Class' é a variável alvo (0 = transação normal, 1 = transação fraudulenta).
- Pasta "imagens", contem imagens que auxiliam no entendimento do conteudo do dataset e no resultado da analise.

## Resultados Iniciais(Random Forest)
- **Acurácia**: 0.9995
- **Precisão para Fraude**: 0.97
- **Recall para Fraude**: 0.76
- **F1-Score para Fraude**: 0.85
- **Matriz de Confusão**:
```
[[85292     3]
 [   36   112]]
```

## Pontos Fortes do Modelo
- Alta acurácia e precisão geral na detecção de fraudes.
- Poucos falsos positivos, o que é desejável para evitar transtornos ao usuário.

## Pontos Fracos do Modelo
- O recall para fraudes pode ser melhorado, já que o modelo não identifica 24% das fraudes.
- Desbalanceamento das classes apresenta um desafio na generalização para detecção de fraudes.

## Resultados Iniciais(Decision Tree)
- **Acurácia**: 0.9995
- **Precisão para Fraude**: 0.97
- **Recall para Fraude**: 0.76
- **F1-Score para Fraude**: 0.85
- **Matriz de Confusão**:
```
[[85292     3]
 [   36   112]]
```

## Pontos Fortes do Modelo


## Pontos Fracos do Modelo


## Resultados Iniciais(Knn)
- **Acurácia**: 0.9995
- **Precisão para Fraude**: 0.97
- **Recall para Fraude**: 0.76
- **F1-Score para Fraude**: 0.85
- **Matriz de Confusão**:
```
[[85292     3]
 [   36   112]]
```

## Pontos Fortes do Modelo


## Pontos Fracos do Modelo


## Resultado final

---



[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download&select=creditcard.csv]: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download&select=creditcard.csv