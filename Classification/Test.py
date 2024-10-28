from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Create SparkSession
spark = SparkSession.builder \
    .appName("FBLiveTH Classification") \
    .getOrCreate()

# Load data file into DataFrame
data = spark.read.csv("fb_live_thailand.csv", header=True, inferSchema=True)

# StringIndexer to create indexes for 'status_type' and 'status_published'
status_type_indexer = StringIndexer(inputCol="status_type", outputCol="status_type_ind")
status_published_indexer = StringIndexer(inputCol="status_published", outputCol="status_published_ind")

# Fit and transform the indexers
data = status_type_indexer.fit(data).transform(data)
data = status_published_indexer.fit(data).transform(data)

# VectorAssembler to combine 'status_type_ind' and 'status_published_ind' into 'features'
assembler = VectorAssembler(inputCols=["status_type_ind", "status_published_ind"], outputCol="features")

# Create Logistic Regression model with parameters
log_reg = LogisticRegression(labelCol="status_type_ind", featuresCol="features", 
                             maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Create a pipeline with assembler and logistic regression model
pipeline = Pipeline(stages=[assembler, log_reg])

# Split the data into train and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Fit the pipeline on the training data
pipeline_model = pipeline.fit(train_data)

# Transform the test data
predictions = pipeline_model.transform(test_data)

# Show 5 rows of the predictions DataFrame
predictions.select("features", "status_type_ind", "prediction").show(5)

# Evaluation using MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="status_type_ind", predictionCol="prediction")

# Calculate and display evaluation metrics
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Measure: {f1}")

# Stop SparkSession
spark.stop()
