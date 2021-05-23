package org.enset.iibdcc

import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.slf4j.LoggerFactory

import scala.util.Random

object BooksRatingsALS {
  final val LOGGER = LoggerFactory.getLogger(BooksRatingsALS.getClass.getName)

  def main(args: Array[String]): Unit = {
    require(args.length > 1, "Error, Arguments not found!")
    /**Configuring the spark environment */
    val jarFile = "target/scala-2.11/bookratingals_2.11-0.1.jar"
    val conf = new SparkConf()
      .setAppName(BooksRatingsALS.getClass.getName).setMaster("local[*]")
      .set("spark.executor.memory", "8g").setJars(Seq(jarFile))
    val sc = new SparkContext(conf)
    val spark = new SparkSession.Builder().appName("test").master("local").getOrCreate()
    /**Extracting Csv Data and Storing them in DataFrames + transform ratings data types from (String,String,String)
     * to (Int,Int,Double) + transform books data types from (String,String,String)
     * to (Int,String,String) + printing Schema and 1st five data rows*/
    //Modifying the path in here to an HDFS Path instead of what is down below:
    // "/src/main/resources/ratings.csv"
    val dfRatings = spark.read.option("header", "true")
      .csv("hdfs://hadoop-master:9000/user/root/data/books-input/ratings.csv")
      .withColumn("book_id", col("book_id").cast(IntegerType))
      .withColumn("user_id",col("user_id").cast(IntegerType))
      .withColumn("rating",col("rating").cast(DoubleType))

    dfRatings.printSchema()
    dfRatings.show(5)
    /**Initial read of Books DataFrame stored in 'dfBooks'*/
      //Modifying the path in here to an HDFS Path instead of what is down below:
      // "/src/main/resources/books.csv"
    val dfBooks = spark.read.option("header","true")
      .option("inferSchema","true")
      .csv("hdfs://hadoop-master:9000/user/root/data/books-input/books.csv")
    dfBooks.printSchema()
    dfBooks.show(5)
    /**Creating a View to select Data from it easily*/
    dfBooks.createOrReplaceTempView("BookTable")
    /**Selecting only valuable data for the users*/
    import spark.implicits._
    val dfBooksToRDD = spark.sql("SELECT id, title, authors FROM BookTable").
      withColumn("id",col("id").cast(IntegerType)).map{ line =>
      val id=line(0).toString
      val title=line(1).toString
//    val author=line(2).toString
      (id.toInt,title)
    }.rdd.collect.toMap

    /**Printing data and Schema from dfBooksSqlQuery*/
//    dfBooksSqlQuery.printSchema()
//    dfBooksSqlQuery.show(10)

    val ratingsRDD = dfRatings.select("book_id","user_id","rating").orderBy("user_id")
      .map { rat =>
        val user_id=rat(1).toString
        val book_id=rat(0).toString
        val rating = rat(2).toString
      (Random.nextInt(10),Rating(user_id.toInt,book_id.toInt,rating.toDouble))
    }.rdd

    /**Counting the number of ratings, users and Books*/
    val numRatings = ratingsRDD.count()
    val numUsers = ratingsRDD.map(_._2.user).distinct.count
    val numBooks = ratingsRDD.map(_._2.product).distinct.count
    println("We got " + numRatings + " ratings from " + numUsers + " users on " + numBooks + " books. ")

    /**To make recommendation for you, we will extract top 50 Books rated then*/
    /**we will ask you to do some rating, later we will return your ratings..*/
    val mostRatedBooksIds = ratingsRDD.map(_._2.product).countByValue().toSeq
      .sortBy(-_._2).take(50).map(_._1)
    val random = new Random(0)
    val selectedBooks = mostRatedBooksIds.filter(x => random.nextDouble() < 0.2)
    .map(x => (x, dfBooksToRDD(x)))

    /** Elicitate ratings from command-line. */
    def elicitateRatings(books: Seq[(Int, String)]) = {
      val prompt = "Rate the following movies (1-5 points)"
      println(prompt)
      val ratings = books.flatMap { x =>
        var rating: Option[Rating] = None
        var valid = false
        while (!valid) {
          print(x._2 + ": ")
          try {
            val r = Console.readInt
            if (r < 0 || r > 5) {
              println(prompt)
            } else {
              valid = true
              if (r > 0) {
                rating = Some(Rating(0, x._1, r))
              }
            }
          } catch {
            case e: Exception => println(prompt)
          }
        }
        rating match {
          case Some(r) => Iterator(r)
          case None => Iterator.empty
        }
      }
      if (ratings.isEmpty) {
        error("No rating provided!")
      } else {
        ratings
      }

    }
    val myRatings = elicitateRatings(selectedBooks)
    val myRatingsRDD = sc.parallelize(myRatings)

    /**Splitting DataSets into 3 subset of Data : "Training Set 60%", "Validation Set 20%"
     * and "Test Set 20%" */
    val numPartitions = 20
    val training = ratingsRDD.filter(x=> x._1 < 6)
      .values.union(myRatingsRDD).repartition(numPartitions).persist
    val validation = ratingsRDD.filter(x=> x._1>=6 && x._1 <8)
      .values.repartition(numPartitions).persist
    val test = ratingsRDD.filter(x=> x._1>=8)
      .values.persist
    //Counting the number of training, validation and test Sets down below :
    val numTraining = training.count()
    val numValidation = validation.count()
    val numTest = test.count()
    println("Training: " + numTraining + " , Validation:" + numValidation + " , Test: " +numTest)

    /** Compute RMSE (Root Mean Squared Error). */
    def computeRMSE(model:MatrixFactorizationModel, data:RDD[Rating], n:Long): Double = {
      val predictions :RDD[Rating] = model.predict(data.map(x=> (x.user,x.product)))
      val predictionsAndRatings = predictions.map(x=>((x.user,x.product),x.rating))
        .join(data.map(x=>((x.user,x.product),x.rating))).values
      math.sqrt(predictionsAndRatings.map(x=>(x._1 - x._2)*(x._1 - x._2)).reduce(_+_)/n)
    }
    /**Training the models by ALS Algorithm + Defining the parameters of ALS */
    val ranks = List(2,4)
    val lambdas = List(0.24, 0.25)
    val numIters = List(10, 20)
    var bestModel:Option[MatrixFactorizationModel] = None
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1
    for(rank <-ranks; lambda<-lambdas; numIter<-numIters){
      val model = ALS.train(training, rank, numIter, lambda)
      val validationRmse = computeRMSE(model, validation, numValidation)
      println("RMSE (Validation) = " + validationRmse + " for the model trained with rank = "
        + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
      if(validationRmse<bestValidationRmse){
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }
    val testRmse = computeRMSE(bestModel.get, test, numTest)
    println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
      + ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRmse +".")

    /**let’s take a look at what books our model recommends for you.*/
    val myRatedBookIds = myRatings.map(_.product).toSet
    val candidates = sc.parallelize(dfBooksToRDD.keys.filter(!myRatedBookIds.contains(_)).toSeq)
    val recommendations  = bestModel.get
      .predict(candidates.map((0, _)))
      .collect
      .sortBy(-_.rating)
      .take(50)


    var i =1
    println("Books recommended for you : ")
    recommendations.foreach{
      r => println("%2d".format(i) + ": " + dfBooksToRDD(r.product))
        i+=1
    }
    /**Does ALS output a non-trivial model? let's see: */
    val meanRating = training.union(validation).map(_.rating).mean
    val baselineRMSE = math.sqrt(test.map(x=>(meanRating - x.rating)
      * (meanRating - x.rating)).reduce(_+_) / numTest)
    val improvement = (baselineRMSE - testRmse) / baselineRMSE*100
    println("The best model improves the baseline by = " + "%1.2f".format(improvement) + "%.")
    sc.stop()
  }//main


}//class
