package itba.poyhenard

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}

trait DatasetUtil {

  // when using console add this
  // implicit val ss = spark
  def loadTrainData(csv: String)(implicit spark: SparkSession) = {
    import org.apache.spark.sql.types._

    val schema = StructType(Seq(
      StructField("default", StringType, nullable = false),
      StructField("limit_bal", DoubleType, nullable = false),
      StructField("sex", DoubleType, nullable = false),
      StructField("education", DoubleType, nullable = false),
      StructField("marriage", DoubleType, nullable = false),
      StructField("age", DoubleType, nullable = false),
      StructField("pay_0", DoubleType, nullable = false),
      StructField("pay_2", DoubleType, nullable = false),
      StructField("pay_3", DoubleType, nullable = false),
      StructField("pay_4", DoubleType, nullable = false),
      StructField("pay_5", DoubleType, nullable = false),
      StructField("pay_6", DoubleType, nullable = false),
      StructField("bill_amt1", DoubleType, nullable = false),
      StructField("bill_amt2", DoubleType, nullable = false),
      StructField("bill_amt3", DoubleType, nullable = false),
      StructField("bill_amt4", DoubleType, nullable = false),
      StructField("bill_amt5", DoubleType, nullable = false),
      StructField("bill_amt6", DoubleType, nullable = false),
      StructField("pay_amt1", DoubleType, nullable = false),
      StructField("pay_amt2", DoubleType, nullable = false),
      StructField("pay_amt3", DoubleType, nullable = false),
      StructField("pay_amt4", DoubleType, nullable = false),
      StructField("pay_amt5", DoubleType, nullable = false),
      StructField("pay_amt6", DoubleType, nullable = false)
    ))

    spark.read.
      option("header", false).
      schema(schema).
      csv(csv)
  }

  def loadUserInputData(csv: String)(implicit spark: SparkSession) = {
    import org.apache.spark.sql.types._
    val schema = StructType(Seq(
      StructField("userId", StringType, nullable = false), // USER ID to identify the PREDICTED ANSWER
      StructField("limit_bal", DoubleType, nullable = false),
      StructField("sex", DoubleType, nullable = false),
      StructField("education", DoubleType, nullable = false),
      StructField("marriage", DoubleType, nullable = false),
      StructField("age", DoubleType, nullable = false),
      StructField("pay_0", DoubleType, nullable = false),
      StructField("pay_2", DoubleType, nullable = false),
      StructField("pay_3", DoubleType, nullable = false),
      StructField("pay_4", DoubleType, nullable = false),
      StructField("pay_5", DoubleType, nullable = false),
      StructField("pay_6", DoubleType, nullable = false),
      StructField("bill_amt1", DoubleType, nullable = false),
      StructField("bill_amt2", DoubleType, nullable = false),
      StructField("bill_amt3", DoubleType, nullable = false),
      StructField("bill_amt4", DoubleType, nullable = false),
      StructField("bill_amt5", DoubleType, nullable = false),
      StructField("bill_amt6", DoubleType, nullable = false),
      StructField("pay_amt1", DoubleType, nullable = false),
      StructField("pay_amt2", DoubleType, nullable = false),
      StructField("pay_amt3", DoubleType, nullable = false),
      StructField("pay_amt4", DoubleType, nullable = false),
      StructField("pay_amt5", DoubleType, nullable = false),
      StructField("pay_amt6", DoubleType, nullable = false)
    ))

    spark.read.
      option("header", false).
      schema(schema).
      csv(csv)
  }

  def vectorizeInput(df: DataFrame)(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._

    val featureCols = Array("limit_bal","sex","education","marriage","age","pay_0",
      "pay_2","pay_3","pay_4","pay_5","pay_6","bill_amt1","bill_amt2","bill_amt3",
      "bill_amt4","bill_amt5","bill_amt6","pay_amt1","pay_amt2","pay_amt3","pay_amt4",
      "pay_amt5","pay_amt6")

    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val out = assembler.transform(df)
    out.select('features).show(truncate = false)

    out
  }
}
