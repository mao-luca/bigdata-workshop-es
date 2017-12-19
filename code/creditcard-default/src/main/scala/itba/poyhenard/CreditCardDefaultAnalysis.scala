package itba.poyhenard

import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.sql.SparkSession

object CreditCardDefaultAnalysis extends DatasetUtil {

  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      System.err.println(
        s"""
           |Usage: CreditRiskAnalysis <datasource> <model>
           |  <datasource> CSV dataset to PREDICT credit
           |  <model> path to the model
           |
           |  CreditRiskAnalysis /dataset/creditcard-default/UCI_Credit_Card_Input.csv /dataset/creditcard-default.model
        """.stripMargin)
      System.exit(1)
    }

    //CAMBIAR CUANDO COMPILE LA CLASE!
    //val Array(datasource, modelPath) = Array("/dataset/creditcard-default/UCI_Credit_Card_Input.csv",
    //  "/dataset/creditcard-default.model")

    val Array(datasource, modelPath) = args

    //    implicit val ss = spark
    implicit val spark = SparkSession.
      builder.
      appName("CreditCardDefault").
      getOrCreate()

    val df = loadUserInputData(datasource)
    val dfVector = vectorizeInput(df)

    val model = RandomForestClassificationModel.load(modelPath)
    val predictions = model.transform(dfVector)

    import spark.implicits._

    println("=" * 30)
    println("Prediction are:")
    predictions.select($"userId", $"prediction").show(false)
  }


}
