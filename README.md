# 📊 Análise de Churn TelecomX Parte 2

# Tecnologias Utilizadas

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-013243.svg)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-F7931E.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-red.svg)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.13+-purple.svg)
![Imbalanced--Learn](https://img.shields.io/badge/Imbalanced--Learn-0.8+-brightgreen.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.6+-darkgreen.svg)

## Técnicas e Algoritmos

![SMOTE](https://img.shields.io/badge/Balanceamento-SMOTE-yellow.svg)
![StandardScaler](https://img.shields.io/badge/Normalização-StandardScaler-lightblue.svg)
![Logistic Regression](https://img.shields.io/badge/Modelo-Logistic%20Regression-blueviolet.svg)
![XGBoost Model](https://img.shields.io/badge/Modelo-XGBoost-success.svg)

## 🎯 Objetivos do Projeto
- Preparar os dados para a modelagem (tratamento, encoding, normalização).
- Realizar análise de correlação e seleção de variáveis.
- Treinar dois ou mais modelos de classificação.
- Avaliar o desempenho dos modelos com métricas.
- Interpretar os resultados, incluindo a importância das variáveis.
- Criar uma conclusão estratégica apontando os principais fatores que influenciam a evasão.

## 🚀 Como executar o projeto

1. Faça o download dos arquivos deste repositório:
   - `Challenge_TelecomX_Parte_2_Lidia_Lapertosa.ipynb`
   - `df_limpo.csv` (arquivo de dados)

2. Abra o **Google Colab** ([link](https://colab.research.google.com/)).

3. Faça o upload dos arquivos para o Colab.

4. No notebook, importe o arquivo CSV utilizando **Pandas**:

   ```python
   import pandas as pd
   df = pd.read_csv("df_limpo.csv")

5. Execute as células na ordem em que estão no notebook para reproduzir toda a análise.

## 📝 Metodologia

O projeto foi desenvolvido em diferentes etapas:

1. **Preparação dos Dados**
   * Tratamento de valores ausentes
   * Padronização de variáveis categóricas e numéricas

2. **Correlação e Seleção de Variáveis**
   * Análise exploratória
   * Identificação das variáveis mais relevantes para o churn

3. **Modelagem Preditiva**
   * Treinamento com **Logistic Regression** e **XGBoost**

4. **Análise dos Modelos**
   * Avaliação usando métricas de classificação: AUC, F1-Score, Precision e Recall

5. **Importância das Variáveis**
   * Análise da contribuição de cada variável em cada modelo
   * Visualização gráfica da importância das features

6. **Conclusão**
   * Principais insights
   * Recomendação final para a empresa

## 🤖 Modelos Utilizados

Foram selecionados dois algoritmos complementares para análise comparativa:

**Logistic Regression**

* Modelo linear probabilístico com alta interpretabilidade
* Coeficientes fornecem insights diretos sobre impacto das features
* Baseline robusto para problemas de classificação binária

**XGBoost (Extreme Gradient Boosting)**

* Ensemble de árvores de decisão com boosting
* Captura interações complexas e relações não-lineares
* Otimização automática de hiperparâmetros e regularização

## 🔍 Comparação de Métricas

| Métrica | Logistic Regression | XGBoost | 🥇 Vencedor |
|---------|-------------------|---------|-------------|
| **Acurácia** | 75.1% | 76.5% | XGBoost |
| **ROC AUC** | 84.5% | 80.9% | **Logistic Regression** |
| **Recall (Churn)** | 81% | 56% | **Logistic Regression** |
| **Precision (Churn)** | 52% | 56% | XGBoost |

## 📊 Principais Variáveis

Os modelos identificaram variáveis com maior impacto no churn.

* **Logistic Regression**

<img width="821" height="590" alt="variaveis_regressao_logistica" src="https://github.com/user-attachments/assets/76fd9119-6d62-4c7a-bd1c-be8d5428c7e6" />

* **XGBoost**
<img width="790" height="590" alt="variaveis_xgboost" src="https://github.com/user-attachments/assets/f9d28101-6625-4da6-8960-a2b1e8c9771e" />

## 📌 Conclusão e Recomendações

* Clientes com **menor tempo de contrato** apresentam maior probabilidade de churn.
* O **tipo de serviço de internet (Fiber Optic)** foi uma variável fortemente associada ao cancelamento.
* **Contratos mais longos (2 anos)** reduzem significativamente o churn.

**Recomendação Final:**
A empresa deve investir em **programas de fidelização** e **melhoria na percepção do serviço de internet fibra**, já que esses fatores são cruciais para retenção.

## 💡 Principais Insights

* O **XGBoost** apresentou maior acurácia geral, mas o **Logistic Regression** se destacou em Recall (essencial em churn, pois minimizar falsos negativos é mais importante).
* Variáveis de **contrato e tempo de permanência** são determinantes para prever churn.
* Há espaço para otimizar campanhas de retenção segmentadas por perfil de cliente.

## Próximos passos
1. Pré-processamento dos dados
- **Testar outras técnicas de balanceamento além do SMOTE**:
  - **SMOTEENN** ou **SMOTETomek** → combina oversampling + limpeza de ruído.
  - **Random Undersampling** → remover clientes não churn em excesso.
  - **Class Weights** → penalizar mais os erros da classe churn sem alterar os dados.
    ```python
    LogisticRegression(class_weight='balanced')
    XGBClassifier(scale_pos_weight=ratio)
    ```
- **Feature Engineering**  
  Criar novas variáveis pode ter mais impacto que trocar de modelo:
  - Tempo médio até cancelamento.
  - Razão entre gasto atual e gasto histórico.
  - Número de interações recentes do cliente.

2. Modelagem
- **Validação cruzada (Cross-Validation, ex. StratifiedKFold)**  
  Evita depender de uma única divisão treino/teste e gera resultados mais confiáveis.

- **Ajuste de hiperparâmetros (Hyperparameter Tuning)**  
  Tanto Logistic Regression quanto XGBoost podem melhorar bastante com `GridSearchCV`, `RandomizedSearchCV` ou **Optuna**.
  - Logistic Regression → parâmetro `C` (regularização), solver.  
  - XGBoost → parâmetros como `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`.

- **Ensembles**  
  Combinar modelos pode melhorar a performance:
  - Stacking ou blending de Logistic Regression + XGBoost.  
  - Une recall alto de um com precisão do outro.

3. Avaliação
- **Métricas focadas no negócio**  
  Como churn é desbalanceado, acurácia pode enganar.  
  - Usar **F1-score** (equilíbrio entre recall e precisão).  
  - Usar **PR AUC (Precision-Recall AUC)** → mais informativo para dados desbalanceados.  

- **Threshold Tuning**  
  Atualmente, o corte padrão `0.5` está sendo usado.  
  - Ajustar o threshold pode aumentar recall sem perder tanto na precisão.  
  - Exemplo: usar `predict_proba` e escolher o ponto ótimo na curva Precision-Recall.
 
## 📞 Contato

<div align="left">
<img src="https://github.com/user-attachments/assets/07594e84-2524-4810-a5dc-58abc9526ca3" alt="imagem autora do projeto" width="150px">

**Lidia Lapertosa**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lidia-lapertosa/)

📧 Em caso de dúvidas, entre em contato!
