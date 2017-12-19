package itba.poyhenard

import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.sql.SparkSession

/**
  * Created by poyhenard on 12/18/17.
  */
object CreditCardDefaultBoostAnalysis extends DatasetUtil {

  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      System.err.println(
        s"""
           |Usage: CreditCardDefaultBoostAnalysis <datasource> <model>
           |  <datasource> CSV dataset to PREDICT credit
           |  <model> path to the model
           |
           |  CreditRiskAnalysis /dataset/creditcard-default/UCI_Credit_Card_Input.csv /dataset/creditcard-default-boost.model
        """.stripMargin)
      System.exit(1)
    }

    //Para correr en consola
    //val Array(datasource, modelPath) = Array("/dataset/creditcard-default/UCI_Credit_Card_Input.csv",
    //  "/dataset/creditcard-default-boost.model")

    val Array(datasource, modelPath) = args

    //    implicit val ss = spark
    implicit val spark = SparkSession.
      builder.
      appName("CreditCardDefaultBoost").
      getOrCreate()

    val df = loadUserInputData(datasource)
    val dfVector = vectorizeInput(df)

    val model = GBTClassificationModel.load(modelPath)
    val predictions = model.transform(dfVector)

    import spark.implicits._

    println("=" * 30)
    println("Prediction are:")
    predictions.select($"userId", $"prediction").show(false)



  }
}
