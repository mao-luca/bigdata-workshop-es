package itba.poyhenard

import CreditCardDefaultAnalysis.vectorizeInput
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

// Heavily inspired on
// https://mapr.com/blog/predicting-loan-credit-risk-using-apache-spark-machine-learning-random-forests/

object CreditCardDefaultTrain extends DatasetUtil {
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println(
        s"""
           |Usage: CreditCardDefaultTrain <datasource> <model>
           |  <datasource> CSV dataset to learn from
           |  <model> path to save model to
           |
           |  CreditCardDefaultTrain /dataset/creditcard-default/UCI_Credit_Card_Train.csv /dataset/creditcard-default.model
        """.stripMargin)
      System.exit(1)
    }

    //Para correr en consola
    //val Array(datasource, modelPath) = Array("/dataset/creditcard-default/UCI_Credit_Card_Train.csv",
    //      "/dataset/creditcard-default.model")

    val Array(datasource, modelPath) = args


    // When using Spark-Shell:
    // implicit val ss = spark
    implicit val spark = SparkSession.
      builder.
      appName("CreditCardDefault").
      getOrCreate()

    import spark.implicits._

    //Cargar, Verificar y Vectorizar Dataset
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

    val classifier = new RandomForestClassifier().
      setImpurity("gini").
      setMaxDepth(3).
      setNumTrees(20).
      setFeatureSubsetStrategy("auto").
      setSeed(44701)

    val model = classifier.fit(trainingData)
    println(model.toDebugString)

    println("=" * 30)
    println("Before pipeline fitting\n")
    val predictions = model.transform(testData)

    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
    val accuracy = evaluator.evaluate(predictions)
    println(f"Accuracy: $accuracy%2.3f")



    // Let's try to do better - Pipeline Fitted Model
    val paramGrid = new ParamGridBuilder().
      addGrid(classifier.maxBins, Array(20, 40)).
      addGrid(classifier.maxDepth, Array(2, 10)).
      addGrid(classifier.numTrees, Array(10, 60)).
      addGrid(classifier.impurity, Array("entropy", "gini")).
      build()

    val steps: Array[PipelineStage] = Array(classifier)
    val pipeline = new Pipeline().setStages(steps)

    val cv = new CrossValidator().
      setEstimator(pipeline).
      setEvaluator(evaluator).
      setEstimatorParamMaps(paramGrid).
      setNumFolds(10)

    val pipelineFittedModel = cv.fit(trainingData)

    val predictions2 = pipelineFittedModel.transform(testData)
    val accuracy2 = evaluator.evaluate(predictions2)
    println("=" * 30)
    println("AFTER pipeline fitting\n")
    println(f"Accuracy: $accuracy2%2.3f")

    val bestModel = pipelineFittedModel.bestModel.asInstanceOf[PipelineModel].stages(0)
    val params = bestModel.extractParamMap

    println(
      s"""
         |The best model found was:
         |${bestModel}
         |
        |Using params:
         |${params}
         |
      """.stripMargin)

    println("=" * 30)
    println("Summary of the 2 Executions\n")

    println("Initial (single) run")
    println(f"Accuracy: $accuracy%2.3f")

    println("Pipeline fitting with GridSearch optimization")
    println(f"Accuracy: $accuracy2%2.3f")

    // Salvar el modelo

    println("Saving model for further application")

    println(model.toDebugString)
    model.write.overwrite().save(modelPath)

    /*
    TODO: save the best model (or both). If this is done then CrediCardDefaultAnalysis should be adapted to apply also a CrossValidatorModel
    if (accuracy > accuracy2) {
      model.write.overwrite().save(modelPath)

    } else {
      pipelineFittedModel.write.overwrite().save(modelPath)

    }
    */
  }

}

