from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import collect_list, array_distinct, explode, split, col

# Step 1: สร้าง SparkSession เพื่อเริ่มใช้งาน PySpark
spark = SparkSession.builder.appName("FPGrowthExample").getOrCreate()

# Step 2: อ่านข้อมูลจากไฟล์ CSV (สมมุติว่าเป็นข้อมูลสินค้าจากร้านขายของชำ)
data = spark.read.csv("groceries_data.csv", header=True, inferSchema=True)

# Step 3: รวมกลุ่มข้อมูลตาม Member_number และรวบรวม itemDescription เป็นรายการสินค้า จาก itemDescription เปลี่ยนใหม่เป็น Items
grouped_data = data.groupBy("Member_number").agg(collect_list("itemDescription").alias("Items"))

# Step 4: แสดงข้อมูลที่ถูกจัดกลุ่ม (truncate=False เพื่อแสดงข้อมูลแบบเต็ม)
grouped_data.show(truncate=False)

# Step 5: เพิ่มคอลัมน์ 'basket' ที่มีรายการสินค้าที่ไม่ซ้ำกัน
# array_distinct() เป็นฟังก์ชันที่ใช้ใน PySpark SQL เพื่อทำให้อาร์เรย์ไม่มีค่าที่ซ้ำกัน 
# โดยเมื่อใช้งาน array_distinct กับคอลัมน์ที่มีค่าที่เป็นอาร์เรย์ ฟังก์ชันนี้จะทำการลบค่าในอาร์เรย์ที่ซ้ำกันออก แล้วคืนค่าอาร์เรย์ใหม่ที่มีเพียงค่าที่ไม่ซ้ำกัน
# ถ้ามีข้อมูลเป็น [1, 2, 2, 3] การใช้ array_distinct จะได้ผลลัพธ์เป็น [1, 2, 3]
grouped_data = grouped_data.withColumn("basket", array_distinct(grouped_data["Items"]))

# Step 6: แสดงข้อมูลอีกครั้ง
grouped_data.show(truncate=False)

# Step 7: แตก (explode) อาร์เรย์ Items เพื่อแยกรายการสินค้าให้อยู่ในรูปแบบของแถว เปลี่ยนใหม่เป็น item
exploded_data = grouped_data.select("Member_number", explode("Items").alias("item"))

# Step 8: แทนที่เครื่องหมาย '/' ด้วย ',' เพื่อแยกประเภทสินค้าที่รวมกันในรายการเดียว
# milk\\eggs\\bread ถ้ามันเจอคำแบบนี้บน ไฟล์ csv มันจะทำการแยกออกจากกันเลย
# separated_data = exploded_data.withColumn("item", explode(split("item", "\\\\")))

separated_data = exploded_data.withColumn("item", explode(split("item", "/")))

# Step 9: รวมรายการสินค้ากลับเข้าไปในรูปแบบอาร์เรย์และให้มั่นใจว่าไม่มีรายการที่ซ้ำกัน
final_data = separated_data.groupBy("Member_number").agg(collect_list("item").alias("Items"))

# Step 10: ทำให้รายการในอาร์เรย์ Items ไม่ซ้ำกัน
final_data = final_data.withColumn("Items", array_distinct(col("Items")))

# Step 11: แสดงข้อมูลที่แยกเรียบร้อยแล้ว
final_data.show(truncate=False)

# Step 12: สร้างโมเดล FPGrowth โดยกำหนดค่า minSupport และ minConfidence
# การ prediction ขึ้นกับ minSupport และ minConfidence
minSupport = 0.1
minConfidence = 0.2

# itemsCol='Items': กำหนดคอลัมน์ที่มีรายการสินค้าที่จะใช้ในการฝึกโมเดล ในที่นี้คือคอลัมน์ Items
# predictionCol='prediction': กำหนดชื่อคอลัมน์ที่จะแสดงผลการทำนาย
fp = FPGrowth(minSupport=minSupport, minConfidence=minConfidence, itemsCol='Items', predictionCol='prediction')


# Step 13: ฝึกโมเดล FPGrowth กับข้อมูลที่แยกเรียบร้อยแล้ว
model = fp.fit(final_data)

# Step 14: แสดง itemsets ที่พบบ่อยในข้อมูล
model.freqItemsets.show(10)  # แสดง itemsets ที่พบบ่อย 10 อันดับแรก

# Step 15: กรองกฎความสัมพันธ์ (association rules) โดยใช้ค่า confidence ที่มากกว่า 0.4
filtered_rules = model.associationRules.filter(model.associationRules.confidence > 0.4)

# Step 16: แสดงกฎความสัมพันธ์ที่ถูกกรอง
filtered_rules.show(truncate=False)

# Step 17: สร้าง DataFrame ใหม่เพื่อใช้ในการทำนาย (prediction)
new_data = spark.createDataFrame(
    [
        (["vegetable juice", "frozen fruits", "packaged fruit"],),
        (["mayonnaise", "butter", "buns"],)
    ],
    ["Items"]  # คอลัมน์ต้องเป็น "Items" เพื่อให้สอดคล้องกับข้อมูลที่ใช้ในโมเดล
)

# Step 18: แสดงข้อมูลใหม่ที่จะใช้ในการทำนาย
new_data.show(truncate=False)

# Step 19: ใช้โมเดลในการทำนายสินค้าที่อาจจะซื้อร่วมกันกับข้อมูลใหม่
predictions = model.transform(new_data)

# Step 20: แสดงผลการทำนาย
predictions.show(truncate=False)

# ปิดการทำงานของ SparkSession
spark.stop()
