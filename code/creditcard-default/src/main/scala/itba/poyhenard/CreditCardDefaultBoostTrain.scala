package itba.poyhenard

import CreditCardDefaultAnalysis.vectorizeInput
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql._
import org.apache.spark.sql.functions._


/**
  * Created by poyhenard on 12/18/17.
  */
object CreditCardDefaultBoostTrain extends DatasetUtil {
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println(
        s"""
           |Usage: CreditCardDefaultBoostTrain <datasource> <model>
           |  <datasource> CSV dataset to learn from
           |  <model> path to save model to
           |
           |  CreditCardDefaultTrain /dataset/creditcard-default/UCI_Credit_Card_Train.csv /dataset/creditcard-default-boost.model
        """.stripMargin)
      System.exit(1)
    }

    //CAMBIAR CUANDO COMPILE LA CLASE!
    //val Array(datasource, modelPath) = Array("/dataset/creditcard-default/UCI_Credit_Card_Train.csv",
    //      "/dataset/creditcard-default-boost.model")

    val Array(datasource, modelPath) = args

    // When using Spark-Shell:
    // implicit val ss = spark
    implicit val spark = SparkSession.
      builder.
      appName("CreditCardDefaultBoost").
      getOrCreate()

    import spark.implicits._

    val creditDF = loadTrainData(datasource)
    creditDF.printSchema
    creditDF.show

    val dfVector = vectorizeInput(creditDF)

    // Convert Strings into Label Identifiers (Double)
    val labelIndexer = new StringIndexer().setInputCol("default").setOutputCol("label")

    // Add Label Identifiers field to the DF
    val dfLabeled = labelIndexer.fit(dfVector).transform(dfVector)
    dfLabeled.select($"features", $"label", $"default").show(30, false)

    val splitSeed = 44701
    val Array(trainingData, testData) = dfLabeled.randomSplit(Array(0.7, 0.3), splitSeed)

    val classifier = new GBTClassifier().
      setMaxDepth(3).
      setMaxIter(10).
      setSeed(44701)


    val model = classifier.fit(trainingData)
    println(model.toDebugString)

    println("=" * 30)
    println("Before pipeline fitting\n")
    val predictions = model.transform(testData)
    predictions.show

    val evaluator = new MulticlassClassificationEvaluator().
    setLabelCol("label").
    setPredictionCol("prediction").
    setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(f"Accuracy: $accuracy%2.3f")
    println("Test Error = " + (1.0 - accuracy))

    model.write.overwrite().save(modelPath)


  }
}
