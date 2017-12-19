# Credit Card Default Analysis
## Spark Machine Learning (Random Forest and GBT)

### Objetivo
El objetivo principal de este ejemplo es construir 1 modelo de clasificación que permita predecir si un usuario de Tarjeta de Crédito entrará o no en default de pagos en el próximo ciclo de facturación. Para ello se implementaron 2 algoritmos de clasificación (Random Forest and GBT) provedos por la librera MLLIB.

### El Dataset
Los datos pertenecen al dataset público UCI_credit_default_prediction. El mismo junto con otros detalles puede encontrarse en mhttps://www.kaggle.com/xpuspus/uci-credit-default-prediction


### Ejecución

#### MODELO 1 - Random Forest
```bash
sbt clean assembly

spark-submit \
  --class es.arjon.CreditRiskTrain \
  --master 'local[*]' \
  target/scala-2.11/credit-risk-analysis-assembly-0.1.jar \
  /dataset/credit-risk/germancredit.csv \
  /dataset/credit-risk.model
    


spark-submit \
  --class es.arjon.CreditRiskAnalysis \
  --master 'local[*]' \
  target/scala-2.11/credit-risk-analysis-assembly-0.1.jar \
  /dataset/credit-risk/germancredit-user-input.csv \
  /dataset/credit-risk.model
```
#### MODELO 2 - GBT

# Acknowledge
The original author of this tutorial is **Carol McDonald <caroljmcdonald@gmail.com>** for the MapR article: [Predicting Loan Credit Risk using Apache Spark Machine Learning Random Forests](https://mapr.com/blog/predicting-loan-credit-risk-using-apache-spark-machine-learning-random-forests/), 

**Gustavo Arjones <arjones>** updated the API version (Spark 2.1.2) and made changes on the code to clarify/reduce duplication.
  
We've made some modifications in order use the new dataset for Credit Card Default Analysis, and also included a new GBT model to compare results from both.
  

