# Spark-Scala-predict-price-of-a-diamond-with-linear-regression-decision-tree-random-forest
Usage of Spark machine learning (Linear Regression, Decision tree, Random forest) to create a model that predicts a price of diamonds on a basis of different features of them. GridSearch is applied to find the best combination of parameters of a model

**Information about the dataset**
- Number of inputs: 53 941
- Number of features: 11
- Source of data: https://www.kaggle.com/shivam2503/diamonds

**Import and start of spark session**
```
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

// To see less warnings
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR) //less warnings pop up


// Start a simple Spark Session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
```

***Import dataset and print schema***

```val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("diamonds.csv")
data.printSchema()```

>|-- _c0: integer (nullable = true)<br />
>|-- carat: double (nullable = true)<br />
>|-- cut: string (nullable = true)<br />
>|-- color: string (nullable = true)<br />
>|-- clarity: string (nullable = true)<br />
>|-- depth: double (nullable = true)<br />
>|-- table: double (nullable = true)<br />
>|-- price: integer (nullable = true)<br />
>|-- x: double (nullable = true)<br />
>|-- y: double (nullable = true)<br />
>|-- z: double (nullable = true)<br />

