from pyspark.sql import SparkSession  # นำเข้า SparkSession ซึ่งเป็นจุดเริ่มต้นในการใช้ PySpark
from pyspark.ml.feature import StringIndexer, VectorAssembler  # นำเข้า StringIndexer และ VectorAssembler
from pyspark.ml.classification import LogisticRegression  # นำเข้า LogisticRegression เพื่อใช้ในการสร้างโมเดลการจำแนกประเภท
from pyspark.ml.evaluation import MulticlassClassificationEvaluator  # นำเข้า Evaluator สำหรับการประเมินผลโมเดล
from pyspark.ml import Pipeline  # นำเข้า Pipeline สำหรับจัดการกระบวนการสร้างโมเดล

# สร้าง SparkSession และตั้งชื่อแอปพลิเคชัน
spark = SparkSession.builder \
    .appName("LogisticRegressionPipelineExample") \
    .getOrCreate()

# อ่านข้อมูลจากไฟล์ CSV
data = spark.read.csv('fb_live_thailand.csv', header=True, inferSchema=True)

# สร้าง StringIndexer สำหรับคอลัมน์ status_type และ status_published
indexer_status_type = StringIndexer(inputCol="status_type", outputCol="status_type_ind")
indexer_status_published = StringIndexer(inputCol="status_published", outputCol="status_published_ind")

# ปรับใช้ StringIndexer กับข้อมูล
data = indexer_status_type.fit(data).transform(data)
data = indexer_status_published.fit(data).transform(data)

# สร้าง VectorAssembler เพื่อรวมฟีเจอร์เป็นเวกเตอร์
assembler = VectorAssembler(
    inputCols=["status_type_ind", "status_published_ind"],
    outputCol="features"
)

# สร้างโมเดล LogisticRegression
log_reg = LogisticRegression(
    labelCol="status_type_ind",    # คอลัมน์ label สำหรับการจำแนกประเภท
    featuresCol="features",        # คอลัมน์ฟีเจอร์
    maxIter=10,                    # จำนวนรอบการฝึก
    regParam=0.1,                  # พารามิเตอร์การปรับปรุง
    elasticNetParam=0.8            # พารามิเตอร์ ElasticNet
)

# สร้าง Pipeline ที่ประกอบด้วย assembler และ log_reg
pipeline = Pipeline(stages=[assembler, log_reg])

# แบ่งข้อมูลเป็นชุดฝึก (80%) และชุดทดสอบ (20%)
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# ฝึกโมเดลด้วยข้อมูลชุดฝึก
pipeline_model = pipeline.fit(train_data)

# สร้างการทำนายจากชุดข้อมูลทดสอบ
predictions = pipeline_model.transform(test_data)

# แสดงผลลัพธ์การทำนาย 5 แถว
predictions.select("status_type_ind", "prediction", "probability", "features").show(5)

# สร้าง Evaluator สำหรับประเมินผลโมเดล
evaluator = MulticlassClassificationEvaluator(
    labelCol="status_type_ind",    # คอลัมน์ label
    predictionCol="prediction"      # คอลัมน์ prediction
)
print()
print('='*50)
print()
# คำนวณความแม่นยำของโมเดล
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
print(f"Accuracy: {accuracy}")

# คำนวณความแม่นยำแบบน้ำหนัก
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
print(f"Precision: {precision}")

# คำนวณความรู้จำแบบน้ำหนัก
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
print(f"Recall: {recall}")

# คำนวณ F1 Score
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
print(f"F1 Score: {f1}")
print()
print('='*50)
print()

# หยุด SparkSession
spark.stop()
