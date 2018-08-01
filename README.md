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

```
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("diamonds.csv")
data.printSchema()
```  

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

**Subset of data**

`data.show`

>+---+-----+---------+-----+-------+-----+-----+-----+----+----+----+<br />
>|_c0|carat|      cut|color|clarity|depth|table|price|   x|   y|   z|<br />
>+---+-----+---------+-----+-------+-----+-----+-----+----+----+----+<br />
>|  1| 0.23|    Ideal|    E|    SI2| 61.5| 55.0|  326|3.95|3.98|2.43|<br />
>|  2| 0.21|  Premium|    E|    SI1| 59.8| 61.0|  326|3.89|3.84|2.31|<br />
>|  3| 0.23|     Good|    E|    VS1| 56.9| 65.0|  327|4.05|4.07|2.31|<br />
>|  4| 0.29|  Premium|    I|    VS2| 62.4| 58.0|  334| 4.2|4.23|2.63|<br />
>|  5| 0.31|     Good|    J|    SI2| 63.3| 58.0|  335|4.34|4.35|2.75|<br />

**Some data preprocessing**

```
//drop column with ids
val df_noid = data.drop(data.col("_c0"))

val df_no_na = df_noid.na.drop()

val df_label = df_no_na.select(data("price").as("label"), $"carat", $"cut", $"color", $"clarity", $"depth", $"table", $"x", $"y", $"z")
```

**Encode categorical variables: convert strings to integers and encode with OneHotEncoderEstimator**

```
// Import VectorAssembler and Vectors
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

val cutIndexer = new StringIndexer().setInputCol("cut").setOutputCol("cutIndex")
val colorIndexer = new StringIndexer().setInputCol("color").setOutputCol("colorIndex")
val clarityIndexer = new StringIndexer().setInputCol("clarity").setOutputCol("clarityIndex")

import org.apache.spark.ml.feature.OneHotEncoderEstimator
val encoder = new OneHotEncoderEstimator().setInputCols(Array("cutIndex", "colorIndex", "clarityIndex")).setOutputCols(Array("cutIndexEnc", "colorIndexEnc", "clarityIndexEnc"))
```

**Vector assembler**

```
val assembler = (new VectorAssembler()
                    .setInputCols(Array("carat", "cutIndexEnc", "colorIndexEnc", "clarityIndexEnc", "depth", "table", "x", "y", "z"))
                    .setOutputCol("features_assem") )
```

**Scalling of features with MinMaxScaler**

```
import org.apache.spark.ml.feature.MinMaxScaler
val scaler = new MinMaxScaler().setInputCol("features_assem").setOutputCol("features")
```

**Train/Test split**

```val Array(training, test) = df_label.randomSplit(Array(0.75, 0.25))```

**Building decision tree, contructing a pipeline and creating a ParamGrid**

```
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor

import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
//Dataframe
val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("features")//.setImpurity("variance")

val pipeline = new Pipeline().setStages(Array(cutIndexer,colorIndexer, clarityIndexer,encoder, assembler,scaler, dt))

val paramGrid = new ParamGridBuilder().addGrid(dt.maxDepth, Array(5, 10, 15, 20, 30)).addGrid(dt.maxBins, Array(10, 20, 30, 50)).build()
```


