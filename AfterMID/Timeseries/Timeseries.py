
# Step 2: Import necessary libraries
import pandas as pd  # สำหรับการจัดการข้อมูล
import numpy as np  # สำหรับการทำงานกับอาเรย์และคณิตศาสตร์
import matplotlib.pyplot as plt  # สำหรับการสร้างกราฟ
from pmdarima import auto_arima  # สำหรับการสร้างโมเดล ARIMA อัตโนมัติ
from pmdarima.arima import ADFTest  # สำหรับการตรวจสอบความเป็นสถานะ

# Step 3: Load your dataset from the specified path
df = pd.read_csv('year_sales.csv')  # โหลดข้อมูลจากไฟล์ CSV

# Step 4: Convert the 'Year' column to datetime and set it as the index
df['Year'] = pd.to_datetime(df['Year'])  # แปลงคอลัมน์ Year เป็น datetime
df.set_index('Year', inplace=True)  # ตั้งคอลัมน์ Year เป็น index ของ DataFrame

# Step 5: Plot the data to see the time series
df.plot()  # แสดงกราฟของข้อมูลเพื่อดูแนวโน้ม
plt.title("Original Time Series")  # ตั้งชื่อกราฟ
plt.show()  # แสดงกราฟ

# Step 6: Perform the ADF test to check for stationarity
adf_test = ADFTest(alpha=0.05)  # สร้างอ็อบเจ็กต์ ADFTest โดยใช้ระดับความเชื่อมั่นที่ 0.05
adf_result = adf_test.should_diff(df['Sales'])  # ตรวจสอบว่าต้องทำการ differencing หรือไม่
print(f"ADF Test Result: {adf_result}")  # แสดงผลการทดสอบ ADF

# Step 7: Split the data into training and testing sets
train_size = int(len(df) * 0.8)  # กำหนดขนาดของชุดข้อมูลฝึกเป็น 80% ของข้อมูลทั้งหมด
train = df[:train_size]  # สร้างชุดข้อมูลฝึก
test = df[train_size:]  # สร้างชุดข้อมูลทดสอบ

# Step 8: Fit the ARIMA model using auto_arima
model = auto_arima(train, start_p=0, d=1,
                   start_q=0, max_p=5, max_d=5, max_q=5,
                   start_P=0, D=1, start_Q=0, max_P=5,
                   max_D=5, max_Q=5, m=12,
                   seasonal=True, error_action='warn',
                   trace=True, suppress_warnings=True,
                   stepwise=True, random_state=20,
                   n_fits=50)  # สร้างโมเดล ARIMA โดยใช้ auto_arima พร้อมพารามิเตอร์ที่ระบุ

# Step 9: Forecast future values
n_periods = len(test)  # จำนวนช่วงเวลาที่จะพยากรณ์เท่ากับขนาดของชุดข้อมูลทดสอบ
forecast = model.predict(n_periods=n_periods)  # ทำการพยากรณ์ค่าต่อไป

# Step 10: Convert forecast into a DataFrame and assign dates from the test set
forecast_df = pd.DataFrame(forecast, index=test.index, columns=['Predicted'])  # สร้าง DataFrame สำหรับผลการพยากรณ์

# Step 11: Plot the training, test, and predicted data
plt.figure(figsize=(10, 6))  # กำหนดขนาดของกราฟ
plt.plot(train, label='Train Data')  # แสดงชุดข้อมูลฝึก
plt.plot(test, label='Test Data')  # แสดงชุดข้อมูลทดสอบ
plt.plot(forecast_df, label='Predicted Data')  # แสดงผลการพยากรณ์
plt.legend(loc='best')  # แสดงตำนานในตำแหน่งที่ดีที่สุด
plt.title('Train, Test, and Predicted Data')  # ตั้งชื่อกราฟ
plt.show()  # แสดงกราฟ
