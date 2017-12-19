# Credit Card Default Analysis
## Spark Machine Learning (Random Forest and GBT)

### Objetivo
El objetivo principal de este ejemplo es construir 1 modelo de clasificación que permita predecir si un usuario de Tarjeta de Crédito entrará o no en default de pagos en el próximo ciclo de facturación. Para ello se implementaron 2 algoritmos de clasificación (Random Forest and GBT) provedos por la librería MLLIB.

### El Dataset
Los datos pertenecen al dataset público UCI_credit_default_prediction. El mismo junto con otros detalles puede encontrarse en https://www.kaggle.com/xpuspus/uci-credit-default-prediction.

Dicho dataset fue partido en dos conjuntos (uno para training y otro que se utiliza para aplicar el modelo posteriormente). Los mismos se encuentran guardados en:
**Training del Modelo:** /dataset/creditcard-default/UCI_Credit_Card_Train.csv
**Input de Datos:** /dataset/creditcard-default/UCI_Credit_Card_Input.csv


### Ejecución

Compilar el código en code/creditcard-default/:
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

## Alumnos

- Oyhenard, Pablo
- Rossi, Andrés
