from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Create SparkSession
spark = SparkSession.builder.appName("FBLiveClassification").getOrCreate()

# Load data file into DataFrame (FBLiveTH)
FBLiveTH = spark.read.csv("fb_live_thailand.csv", header=True, inferSchema=True)

# Use StringIndexer to prepare data where the columns status_type and status_published are used as inputs to create indexes
status_type_indexer = StringIndexer(inputCol="status_type", outputCol="status_type_ind")
status_published_indexer = StringIndexer(inputCol="status_published", outputCol="status_published_ind")

# Use OneHotEncoder to create Boolean flag of status_type_ind and status_published_ind
status_type_encoder = OneHotEncoder(inputCol="status_type_ind", outputCol="status_type_encoded")
status_published_encoder = OneHotEncoder(inputCol="status_published_ind", outputCol="status_published_encoded")

# Use VectorAssembler to create vector of encoded status_type_ind and status_published_ind resulting in the column features
vector_assembler = VectorAssembler(
    inputCols=["status_type_encoded", "status_published_encoded"], 
    outputCol="features"
)

# Create pipeline where the stages include output from string indexer, encoder, and vector assembler
pipeline = Pipeline(stages=[
    status_type_indexer, 
    status_published_indexer, 
    status_type_encoder, 
    status_published_encoder, 
    vector_assembler
])

# Fit DataFrame into the created pipeline to create the pipeline model
pipeline_model = pipeline.fit(FBLiveTH)

# Use the created pipeline model to transform the DataFrame data resulting in another DataFrame
data_transformed = pipeline_model.transform(FBLiveTH)

# Create train and test datasets using randomSplit function where the output from the previous step is used as an input
train_data, test_data = data_transformed.randomSplit([0.8, 0.2], seed=1234)

# Create decision tree classification where the status_type_ind is used as a label and the output is features
dt_classifier = DecisionTreeClassifier(labelCol="status_type_ind", featuresCol="features")

# Fit train data into the created decision tree to create the model
dt_model = dt_classifier.fit(train_data)

# Use the created model to transform the test data resulting in a prediction DataFrame
predictions = dt_model.transform(test_data)

# Use MulticlassClassificationEvaluator to create the evaluator
# Use status_type_ind as the label column and prediction column as the prediction column
evaluator = MulticlassClassificationEvaluator(
    labelCol="status_type_ind", 
    predictionCol="prediction"
)

# Show accuracy, precision, recall (metricName:"weightedRecall"), and F1 measure (metricName:"f1")
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
weighted_precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
weighted_recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

print(f"Accuracy: {accuracy}")
print(f"Weighted Precision: {weighted_precision}")
print(f"Weighted Recall: {weighted_recall}")
print(f"F1 Measure: {f1}")

# Show the Test Error where it is calculated as 1.0 - accuracy
test_error = 1.0 - accuracy
print(f"Test Error: {test_error}")

# Stop SparkSession
spark.stop()
