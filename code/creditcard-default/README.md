# Credit Card Default Analysis
## Spark Machine Learning (Random Forest and GBT)

### Objetivo
El objetivo principal de este ejemplo es construir 1 modelo de clasificación que permita predecir si un usuario de Tarjeta de Crédito entrará o no en default de pagos en el próximo ciclo de facturación. Para ello se implementaron 2 algoritmos de clasificación (Random Forest and GBT) provedos por la librera MLLIB.

### El Dataset
Los datos pertenecen al dataset público UCI_credit_default_prediction. El mismo junto con otros detalles puede encontrarse en mhttps://www.kaggle.com/xpuspus/uci-credit-default-prediction


### Ejecución

Compilar el código:
```bash
sbt clean assembly
```

#### MODELO 1 - Random Forest

Crear el modelo y entrenar:
```bash
  spark-submit \
    --class itba.poyhenard.CreditCardDefaultTrain \
    --master 'local[*]' \
    target/scala-2.11/credit-card-default-predictor-assembly-0.1.jar \
    /dataset/creditcard-default/UCI_Credit_Card_Train.csv \
    /dataset/creditcard-default.model
```

Aplicar el modelo al conjunto de datos de INPUT:
```bash
  spark-submit \
    --class itba.poyhenard.CreditCardDefaultAnalysis \
    --master 'local[*]' \
    target/scala-2.11/credit-card-default-predictor-assembly-0.1.jar \
    /dataset/creditcard-default/UCI_Credit_Card_Input.csv \
    /dataset/creditcard-default.model
```

#### MODELO 2 - GBT

Crear el modelo y entrenar:
```bash
  spark-submit \
    --class itba.poyhenard.CreditCardDefaultBoostTrain \
    --master 'local[*]' \
    target/scala-2.11/credit-card-default-predictor-assembly-0.1.jar \
    /dataset/creditcard-default/UCI_Credit_Card_Train.csv \
    /dataset/creditcard-default-boost.model
```

Aplicar el modelo al conjunto de datos de INPUT:
```bash
  spark-submit \
    --class itba.poyhenard.CreditCardDefaultBoostAnalysis \
    --master 'local[*]' \
    target/scala-2.11/credit-card-default-predictor-assembly-0.1.jar \
    /dataset/creditcard-default/UCI_Credit_Card_Input.csv \
    /dataset/creditcard-default-boost.model
```

# Acknowledge
The original author of this tutorial is **Carol McDonald <caroljmcdonald@gmail.com>** for the MapR article: [Predicting Loan Credit Risk using Apache Spark Machine Learning Random Forests](https://mapr.com/blog/predicting-loan-credit-risk-using-apache-spark-machine-learning-random-forests/), 

**Gustavo Arjones <arjones>** updated the API version (Spark 2.1.2) and made changes on the code to clarify/reduce duplication.
  
We've made some modifications in order use the new dataset for Credit Card Default Analysis, and also included a new GBT model to compare results from both.
  

