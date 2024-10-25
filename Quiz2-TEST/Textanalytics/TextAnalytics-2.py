# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import trim, col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create SparkSession
spark = SparkSession.builder \
    .appName("Text Analysis Implementation") \
    .getOrCreate()

# Load data from CSV file (update this with the correct file path)
data = spark.read.csv("reviews_rated.csv", header=True, inferSchema=True)

# Select and preprocess the required columns
data = data.select(trim(data['Review Text']).alias('ReviewText'),
                   data['Rating'].cast(IntegerType()).alias('Rating'))

# Filter out rows where 'ReviewText' or 'Rating' is NULL or NaN
data = data.filter((col("ReviewText").isNotNull()) & (col("Rating").isNotNull()))

# Show the cleaned data
data.show(5)

# Create Tokenizer
tokenizer = Tokenizer(inputCol="ReviewText", outputCol="ReviewTextWords")

# Create StopWordsRemover
stop_word_remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="MeaningfulWords")

# Create HashingTF
hashing_tf = HashingTF(inputCol=stop_word_remover.getOutputCol(), outputCol="features")

# Create a pipeline with tokenizer, stop words remover, and hashingTF
pipeline = Pipeline(stages=[tokenizer, stop_word_remover, hashing_tf])

# Split the data into train and test datasets (80-20 split)
train_data, test_data = data.randomSplit([0.8, 0.2], seed=123)

# Show the train dataset
train_data.show(5)

# Fit the pipeline with the train data
pipeline_model = pipeline.fit(train_data)

# Transform the train and test datasets
train_transformed = pipeline_model.transform(train_data)
test_transformed = pipeline_model.transform(test_data)

# Show the transformed train dataset
train_transformed.show(5)

# Create LogisticRegression model
lr = LogisticRegression(labelCol="Rating", featuresCol="features")

# Fit the model to the transformed train dataset
lr_model = lr.fit(train_transformed)

# Transform the test dataset using the trained LogisticRegression model
predictions = lr_model.transform(test_transformed)

# Show the meaningful words, label, and prediction columns
predictions.select("MeaningfulWords", "Rating", "prediction").show(5)

# Create MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="Rating", predictionCol="prediction", metricName="accuracy")

# Calculate the accuracy
accuracy = evaluator.evaluate(predictions)

# Show the accuracy
print(f"Test Accuracy = {accuracy}")