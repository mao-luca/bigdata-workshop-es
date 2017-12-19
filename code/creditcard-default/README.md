# Credit Card Default Analysis
## Spark Machine Learning (Random Forest and GBT)

### The Dataset
The main goal of this example is to build 2 models using Random Forest and GBT algorithms

https://www.kaggle.com/xpuspus/uci-credit-default-prediction



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

# Acknowledge
The original author of this tutorial is **Carol McDonald <caroljmcdonald@gmail.com>** for the MapR article: [Predicting Loan Credit Risk using Apache Spark Machine Learning Random Forests](https://mapr.com/blog/predicting-loan-credit-risk-using-apache-spark-machine-learning-random-forests/), 

**Gustavo Arjones <arjones>** updated the API version (Spark 2.1.2) and made changes on the code to clarify/reduce duplication.
  
We've made some modifications in order use the new dataset for Credit Card Default Analysis, and also included a new GBT model to compare results from both.
  

