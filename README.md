# Inteligência Artificial 
- Aluno: Douglas Cristiano da Silva 
- RA: 152211260

# Projeto Final A3: Desenvolvimento de um Agente Inteligente

## Instalação de libs
pip install seaborn pandas matplotlib

## Dataset "Credit Card Fraud Detection"
Baixe o dataset disponível [aqui](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download&select=creditcard.csv) e mova-o para a pasta dados.

## Execução do projeto
python .\creditcard_analyzer.py

```
Escolha uma opção:
1 - Rodar o modelo k-NN
2 - Rodar o modelo Decision Tree
3 - Rodar o modelo Random Forest
4 - Rodar o modelo Regressão Logística
5 - Rodar o modelo SVM
6 - Rodar o modelo Gradient Boosting
7 - Rodar todos os modelos
0 - Sair
```

## Estrutura do projeto
```
src
    dados - contem o dataset utilizado no projeto.
    imagens - contem imagens geradas pelos modelos.
    modelos - contem os scripts dos modelos utilizados para as analises
creditcard_analyzer.py - script principal do projeto
README.md
```

## Definição do Problema
O objetivo deste projeto é identificar transações fraudulentas no conjunto de dados "Credit Card Fraud Detection", utilizando varios modelos de analise e identificando, qual obtem o melhor resultado.

## Análise dos Dados
- O dataset contém 284807 transações, das quais apenas 0.17% são fraudulentas.
- As features V1 a V28 foram obtidas a partir de uma transformação PCA das features originais, devido a questões de privacidade e segurança.
- A variável 'Amount' representa o valor da transação, e 'Class' é a variável alvo (0 = transação normal, 1 = transação fraudulenta).
- Pasta "imagens", contem imagens que auxiliam no entendimento do conteudo do dataset e no resultado da analise.

## Desafios Encontrados

- Dataset bem definido e conhecido, resultou em uma acuracia relavantivamente alta em todos os modelos.
- Tempo relevante ao rodar o script dos modelos. Mesmo sendo um dataset não muito grande, os recursos para rodar os modelos podem ser consideráveis.
- Além da **Acuracia** em sí, foi necessário o **Recall** e **F1-Score** para ter um melhor entedimento da analise.
- Variáveis transformadas por PCA, como `V1` a `V28`, foram normalizadas, e novas features, como `log_Amount` e `hour_of_day`, foram criadas para melhorar a separação entre classes. 
- Algoritmos como **Random Forest** e **SVM** se destacaram, mas demandaram ajustes de hiperparâmetros com técnicas como **Randomized Search** para alcançar o desempenho ideal. 

## Resultados Random Forest
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

### Resultados Decision Tree
- **Acurácia**: 0.9987
- **Precisão para Fraude**: 0.64
- **Recall para Fraude**: 0.44
- **F1-Score para Fraude**: 0.52
- **Matriz de Confusão**:
```
[[85273 22] 
[ 83 65]]
```

#### Pontos Fortes do Modelo
- Modelo rápido para treinar e interpretar.
- Apresenta bom desempenho geral na classificação de transações normais.

#### Pontos Fracos do Modelo
- Recall baixo para fraudes, indicando que muitas fraudes não são detectadas.
- Pode ter problemas de overfitting se não for ajustado adequadamente.

---

### Resultados K-NN
- **Acurácia**: 0.9989
- **Precisão para Fraude**: 0.76
- **Recall para Fraude**: 0.54
- **F1-Score para Fraude**: 0.63
- **Matriz de Confusão**:
```
[[85281 14]
[ 69 79]]
```
#### Pontos Fortes do Modelo
- Boa precisão e equilíbrio entre precisão e recall para fraudes.
- Simplicidade de implementação, ideal para conjuntos de dados de tamanho moderado.

#### Pontos Fracos do Modelo
- Desempenho pode degradar em conjuntos de dados maiores ou com alta dimensionalidade.
- Requer ajustes adequados de hiperparâmetros, como o número de vizinhos.

---

### Resultados Gradient Boosting
- **Acurácia**: 0.9984
- **Precisão para Fraude**: 0.73
- **Recall para Fraude**: 0.16
- **F1-Score para Fraude**: 0.27
- **Matriz de Confusão**:
```
[[85286 9] 
[ 124 24]]
```

#### Pontos Fortes do Modelo
- Modelo robusto em prever transações normais com alta precisão e recall.
- Implementação eficiente para grandes volumes de dados.

#### Pontos Fracos do Modelo
- O desempenho em detectar fraudes é fraco, com recall significativamente baixo.
- A distribuição desbalanceada afeta a identificação de fraudes.

---

### Resultados Logistic Regression
- **Acurácia**: 0.9991
- **Precisão para Fraude**: 0.85
- **Recall para Fraude**: 0.60
- **F1-Score para Fraude**: 0.70
- **Matriz de Confusão**:
```
[[85279 16] 
[ 59 89]]
```

#### Pontos Fortes do Modelo
- Boa precisão para fraudes, reduzindo falsos positivos.
- Simplicidade e interpretabilidade do modelo.

#### Pontos Fracos do Modelo
- Recall moderado para fraudes, resultando em algumas fraudes não detectadas.
- Pode não capturar relações complexas nos dados.

---

### Resultados SVM
- **Acurácia**: 0.9993
- **Precisão para Fraude**: 0.97
- **Recall para Fraude**: 0.61
- **F1-Score para Fraude**: 0.75
- **Matriz de Confusão**:
```
[[85292 3] 
[ 58 90]]
```

#### Pontos Fortes do Modelo
- Alta precisão e F1-Score para fraudes, indicando bom equilíbrio entre precisão e recall.
- Eficiente para conjuntos de dados de alta dimensionalidade.

#### Pontos Fracos do Modelo
- Recall ainda pode ser aprimorado para identificar mais fraudes.
- Tempo de treinamento mais alto em comparação com modelos lineares.

## Conclusões
A combinação das melhorias implementadas e a diversificação dos algoritmos testados permitiu identificar modelos com melhor desempenho para a detecção de fraudes:

- Random Forest e SVM foram os melhores modelos, equilibrando precisão e recall.

- Modelos como Decision Tree e Logistic Regression mostraram-se úteis como baselines, mas exigem ajustes adicionais para melhorar o recall.

- Gradient Boosting, apesar de seu desempenho limitado na detecção de fraudes, oferece insights sobre a robustez em conjuntos de dados desbalanceados.

- Esses resultados destacam a importância de ajustar tanto os dados quanto os hiperparâmetros, além de considerar diferentes abordagens para atender aos requisitos específicos do problema. 

- Com as melhorias implementadas, foi possível obter modelos mais eficazes e confiáveis na detecção de fraudes, equilibrando as necessidades de precisão e recall, essenciais para aplicações reais.

---