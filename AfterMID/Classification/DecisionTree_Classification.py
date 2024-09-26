from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("DecisionTree_Classification").getOrCreate()

data = spark.read.csv("fb_live_thailand.csv", header=True, inferSchema=True)
# โหลดข้อมูลจากไฟล์ CSV เข้าสู่ DataFrame โดยมีแถวแรกเป็นชื่อคอลัมน์ และให้ Spark ตรวจจับประเภทข้อมูลอัตโนมัติ

status_type_indexer = StringIndexer(inputCol="status_type", outputCol="status_type_ind")
# สร้าง StringIndexer เพื่อแปลงคอลัมน์ 'status_type' เป็นดัชนีใหม่ 'status_type_ind'

status_published_indexer = StringIndexer(inputCol="status_published", outputCol="status_published_ind")
# สร้าง StringIndexer เพื่อแปลงคอลัมน์ 'status_published' เป็นดัชนีใหม่ 'status_published_ind'

status_type_encoder = OneHotEncoder(inputCol="status_type_ind", outputCol="status_type_encoded")
# สร้าง OneHotEncoder เพื่อแปลงดัชนี 'status_type_ind' เป็นข้อมูล Boolean ในคอลัมน์ 'status_type_encoded'

status_published_encoder = OneHotEncoder(inputCol="status_published_ind", outputCol="status_published_encoded")
# สร้าง OneHotEncoder เพื่อแปลงดัชนี 'status_published_ind' เป็นข้อมูล Boolean ในคอลัมน์ 'status_published_encoded'

assembler = VectorAssembler(inputCols=["status_type_encoded", "status_published_encoded"], outputCol="features")
# รวมฟีเจอร์ที่เข้ารหัสไว้ในเวกเตอร์ในคอลัมน์ 'features'

pipeline = Pipeline(stages=[status_type_indexer, status_published_indexer, status_type_encoder, status_published_encoder, assembler])
# สร้าง pipeline ที่รวมทุกขั้นตอนที่ได้ทำไว้ในรูปแบบลำดับขั้นตอน

pipeline_model = pipeline.fit(data)
# ฟิตข้อมูลเข้าสู่ pipeline เพื่อสร้างโมเดล pipeline

transformed_data = pipeline_model.transform(data)
# ใช้ pipeline model เพื่อแปลงข้อมูลและสร้าง DataFrame ใหม่ที่มีฟีเจอร์ใหม่

train_data, test_data = transformed_data.randomSplit([0.8, 0.2])
# แบ่งข้อมูลที่แปลงแล้วออกเป็นชุดการฝึก (80%) และชุดทดสอบ (20%)

decision_tree = DecisionTreeClassifier(labelCol="status_type_ind", featuresCol="features")
# สร้างโมเดล Decision Tree โดยใช้ 'status_type_ind' เป็นคอลัมน์เป้าหมายและ 'features' เป็นฟีเจอร์

decision_tree_model = decision_tree.fit(train_data)
# ฟิตโมเดลด้วยชุดข้อมูลการฝึก

predictions = decision_tree_model.transform(test_data)
# ใช้โมเดลที่ฟิตแล้วเพื่อทำการทำนายชุดข้อมูลทดสอบ

evaluator = MulticlassClassificationEvaluator(labelCol="status_type_ind", predictionCol="prediction")
# สร้าง evaluator สำหรับประเมินผลลัพธ์การทำนายโดยใช้คอลัมน์เป้าหมายและคอลัมน์การทำนาย

accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
# ประเมินความแม่นยำของโมเดล

precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
# ประเมินความแม่นยำเชิงสัมพันธ์ของโมเดล

recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
# ประเมินการเรียกคืนของโมเดล

f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
# ประเมินค่า F1 ของโมเดล

test_error = 1.0 - accuracy
# คำนวณค่า Test Error โดยการลบความแม่นยำจาก 1

print(f"Accuracy: {accuracy}")
# แสดงผลความแม่นยำ

print(f"Precision: {precision}")
# แสดงผลความแม่นยำเชิงสัมพันธ์

print(f"Recall: {recall}")
# แสดงผลการเรียกคืน

print(f"F1 Measure: {f1}")
# แสดงผลค่า F1

print(f"Test Error: {test_error}")
# แสดงผลค่า Test Error

spark.stop()
