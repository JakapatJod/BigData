from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("DecisionTreeRegression").getOrCreate()

df = spark.read.format("csv").option("header", True).load("fb_live_thailand.csv")

df = df.select(df.num_reactions.cast("Double"), df.num_loves.cast("Double"))

indexer_reactions = StringIndexer(inputCol="num_reactions", outputCol="num_reactions_ind")
indexer_loves = StringIndexer(inputCol="num_loves", outputCol="num_loves_ind")

encoder_reactions = OneHotEncoder(inputCol="num_reactions_ind", outputCol="reactions_encoded")
encoder_loves = OneHotEncoder(inputCol="num_loves_ind", outputCol="loves_encoded")

vec_assembler = VectorAssembler(inputCols=["reactions_encoded", "loves_encoded"], outputCol="features")

pipeline = Pipeline(stages=[indexer_reactions, indexer_loves, encoder_reactions, encoder_loves, vec_assembler])

pipeline_model = pipeline.fit(df)

df_transformed = pipeline_model.transform(df)

train_df, test_df = df_transformed.randomSplit([0.8, 0.2], seed=1234)

dt_regressor = DecisionTreeRegressor(featuresCol="features", labelCol="num_loves_ind")

dt_model = dt_regressor.fit(train_df)

predictions = dt_model.transform(test_df)

evaluator = RegressionEvaluator(labelCol="num_loves_ind", predictionCol="prediction")

r2 = evaluator.setMetricName("r2").evaluate(predictions)
print(f"R2: {r2}")
