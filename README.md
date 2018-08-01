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

## Decision Tree

**Building a decision tree, contructing a pipeline and creating a ParamGrid**

Parameters: Max depth(5, 10, 15, 20, 30) and Max Bins(10, 20, 30, 50)

```
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor

import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("features")//.setImpurity("variance")

val pipeline = new Pipeline().setStages(Array(cutIndexer,colorIndexer, clarityIndexer,encoder, assembler,scaler, dt))

val paramGrid = new ParamGridBuilder().addGrid(dt.maxDepth, Array(5, 10, 15, 20, 30)).addGrid(dt.maxBins, Array(10, 20, 30, 50)).build()
```

**Cross-validation (3 splits); Predict test data**

```
val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new RegressionEvaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3)
val cvModel = cv.fit(training)
val predictions = cvModel.transform(test)
```

**Evaluate a model**

```
// Select (prediction, true label) and compute test error.
val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

// Select (prediction, true label) and compute test error.
val evaluator_r2 = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("r2")
val r2 = evaluator_r2.evaluate(predictions)
println("R-squared (r^2) on test data = " + r2)

// Select (prediction, true label) and compute test error.
val evaluator_mae = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("mae")
val mae = evaluator_mae.evaluate(predictions)
println("Mean Absolute Error (MAE) on test data = " + mae)

// Select (prediction, true label) and compute test error.
val evaluator_mse = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("mse")
val mse = evaluator_mse.evaluate(predictions)
println("Mean Squared Error (MSE) on test data = " + mse)
predictions.select("features", "label", "prediction").show()
```

>Root Mean Squared Error (RMSE) on test data = 839.790709763866 <br />
>R-squared (r^2) on test data = 0.9556915131409848 <br />
>Mean Absolute Error (MAE) on test data = 381.5670094175047 <br />
>Mean Squared Error (MSE) on test data  = 705248.4362056978 <br />

### Predictions of decision tree model

>+-----+------------------+<br />
>|label|        prediction|<br />
>+-----+------------------+<br />
>|  326|             724.0|<br />
>|  334|            445.74|<br />
>|  337|             362.0|<br />
>|  337|             360.0|<br />
>|  340| 445|<br />
>|  344|448|<br />
>|  357|            445.74|<br />
>|  357|            445.74|<br />
>|  357|388.27|<br />
>|  360|             371.5|<br />
>|  361| 564.26|<br />
>|  362|            445.74|<br />
>|  363|             385.0|<br />
>|  363|435.57|<br />
>|  365| 533.22|<br />
>|  367|             625.0|<br />
>|  367|            445.74|<br />
>|  367|            445.74|<br />
>|  367|            445.74|<br />
>|  367|            445.74|<br />
>+-----+------------------+<br />



## Random Forest

**Building a random forest, contructing a pipeline and creating a ParamGrid**

Parameters to tune: Max Depth (5, 10, 15, 20, 30, 50), Max Bins (10, 20, 30, 50), Number of trees (10, 20).

```
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

val rf = new RandomForestRegressor().setLabelCol("label").setFeaturesCol("features")//.setImpurity("variance")

val pipeline = new Pipeline().setStages(Array(cutIndexer,colorIndexer, clarityIndexer,encoder, assembler,scaler, rf))

val paramGrid = new ParamGridBuilder().addGrid(rf.maxDepth, Array(5, 10, 15, 20, 30, 50)).addGrid(rf.maxBins, Array(10, 20, 30, 50)).addGrid(rf.numTrees, Array(10, 20)).build()
```

**Cross-validation (3 splits); Predict test data**

```
val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new RegressionEvaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3)
val cvModel = cv.fit(training)
val predictions = cvModel.transform(test)
```

**Evaluate a model**

```
// Select (prediction, true label) and compute test error.
val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

// Select (prediction, true label) and compute test error.
val evaluator_r2 = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("r2")
val r2 = evaluator_r2.evaluate(predictions)
println("R-squared (r^2) on test data = " + r2)

// Select (prediction, true label) and compute test error.
val evaluator_mae = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("mae")
val mae = evaluator_mae.evaluate(predictions)
println("Mean Absolute Error (MAE) on test data = " + mae)

// Select (prediction, true label) and compute test error.
val evaluator_mse = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("mse")
val mse = evaluator_mse.evaluate(predictions)
println("Mean Squared Error (MSE) on test data = " + mse)
predictions.select("features", "label", "prediction").show()
```
