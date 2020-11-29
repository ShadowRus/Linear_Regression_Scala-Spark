package org.apache.spark.ml.made

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.made.Spark._sqlc
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}

trait Spark {
  lazy val spark: SparkSession = Spark._spark
  lazy val sqlc: SQLContext = Spark._sqlc

  lazy val schema: StructType = new StructType()
    .add("RM", DoubleType)
    .add("LSTAT", DoubleType)
    .add("PTRATIO", DoubleType)
    .add("MEDV", DoubleType)

  lazy val test_dataset_path: String = getClass.getResource("/Boston.csv").getPath

  lazy val df_raw: DataFrame = _sqlc.read
    .option("header", "true")
    .schema(schema)
    .csv(test_dataset_path)

  lazy val assembler: VectorAssembler = new VectorAssembler()
    .setInputCols(Array("RM","LSTAT","PTRATIO"))
    .setOutputCol("FE")

  lazy val df: DataFrame = assembler
    .transform(df_raw)
    .drop("RM","LSTAT","PTRATIO")
}

object Spark {
  lazy val _spark: SparkSession = SparkSession.builder
    .appName("Simple Application")
    .master("local[4]")
    .getOrCreate()

  lazy val _sqlc: SQLContext = _spark.sqlContext
}