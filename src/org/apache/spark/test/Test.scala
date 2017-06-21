package org.apache.spark.test

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.StringIndexer

object Test {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Simple Application")
    val sc = new SparkContext(conf)
      
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    //KMEANS
    val npart = 216
    
    def time[A](a: => A) = {
    	val now = System.nanoTime
    	val result = a
    	val sec = (System.nanoTime - now) * 1e-9
    	println("Total time (secs): " + sec)
    	result
    }
    
    val file = "hdfs://hadoop-master:8020/user/spark/datasets/higgs/HIGGS.csv"
    val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "false")
      .option("inferSchema", "true").load(file).repartition(npart)
    
    
    import org.apache.spark.ml.feature.VectorAssembler 
    val featureAssembler = new VectorAssembler().setInputCols(df.columns.drop(1)).setOutputCol("features")
    val processedDf = featureAssembler.transform(df).cache()
    
    print("Num. elements: " + processedDf.count)
    
    // Trains a k-means model.
    import org.apache.spark.ml.clustering.KMeans
    val kmeans = new KMeans().setSeed(1L)
    val cmodel = time(kmeans.fit(processedDf.select("features")))    
    
    //RANDOM FOREST
    import org.apache.spark.ml.classification.RandomForestClassifier
    val labelCol = df.columns.head
    
    val indexer = new StringIndexer().setInputCol(labelCol).setOutputCol("labelIndexed")
    val imodel = indexer.fit(processedDf)
    val indexedDF = imodel.transform(processedDf)
    
    val rf = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("labelIndexed")
    val model = time(rf.fit(indexedDF))
  }
}