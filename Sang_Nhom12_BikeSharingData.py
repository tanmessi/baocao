# Reading the dataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import time
#đọc dữ liệu
df = pd.read_csv("day.csv")
print(df.shape)
print(df.head())
# Kiểu dữ liệu mỗi cột
print(df.dtypes)
print("\n")
#Lựa chọn các thuộc tính, df.drop ở đây là loại bỏ các thuộc tính được liệt kê ở đây
df = df.drop(columns=['dteday','instant','season','yr','weathersit','casual','registered'])
#show các thuộc tính còn lại
df.hist(figsize=(12,10))
plt.show()

#Cnt vs Temp
plt.scatter(df['temp'],df['cnt'])
plt.suptitle('Số lượng xe được thuê với nhiệt độ')
plt.xlabel('Nhiệt độ (Temp)')
plt.ylabel('Số lượng xe đạp được thuê (Cnt)')
plt.show()

#Cnt vs atemp
plt.scatter(df['atemp'], df['cnt'])
plt.suptitle('Số lượng xe được thuê với nhiệt độ không khí')
plt.xlabel('Nhiệt độ không khí (Atemp)')
plt.ylabel('Số lượng xe đạp được thuê (Cnt)')
plt.show()

#Cnt vs hum
plt.scatter(df['hum'], df['cnt'])
plt.suptitle('Số lượng xe được thuê với độ ẩm')
plt.xlabel('Độ ẩm (Hum)')
plt.ylabel('Số lượng xe đạp được thuê (Cnt)')
plt.show()

#cnt vs windspedd
plt.scatter(df['windspeed'], df['cnt'])
plt.suptitle('Số lượng xe được thuê với tốc độ gió')
plt.xlabel('Tốc độ gió (Windspeed)')
plt.ylabel('Số lượng xe đạp được thuê (Cnt)')
plt.show()

# Normalizing the data
from sklearn import preprocessing
#gán x = dataframe nhưng loại bỏ cnt vì đó là nhãn
x=df.drop(['cnt'],axis=1)
print(x)
#gán nhãn cho y = cnt
y=df['cnt']
print(y)
x = preprocessing.normalize(x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#Phân chia tập dữ liệu
from sklearn.model_selection import train_test_split
start_time = time.time()
for i in range(100):
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = i)
	#Hồi quy tuyến tính	
	linearRegressor = LinearRegression()
	linearRegressor.fit(x_train, y_train)
	y_predicted = linearRegressor.predict(x_test)
end_time = time.time()
mse = mean_squared_error(y_test, y_predicted)
rmse = np.sqrt(mse)
acc0 = linearRegressor.score(x_test, y_test)*100
mae = mean_absolute_error(y_test,y_predicted)
print("Hoi quy tuyen tinh")
print("Mean Squared Error:",mse)
print("Root Mean Squared Error:", rmse)
print("Do chinh xac:",acc0)
print("Mean Absolute Error:",mae)
print ("Thoi Gian Thuc Thi: %.4f (s)" % (end_time - start_time))
print("\n")
plt.title("Hồi quy tuyến tính")
plt.scatter(y_test,y_predicted)
plt.xlabel("Test")
plt.ylabel("Predict")
plt.grid(True)
plt.show()

# Cây quyết định
from sklearn.tree import DecisionTreeRegressor
start_time = time.time()
for i in range(100):
	regressor = DecisionTreeRegressor(random_state = i)  
	regressor.fit(x_train, y_train) 
	y_predicted_d = regressor.predict(x_test)
end_time = time.time()
mse = mean_squared_error(y_test, y_predicted_d)
rmse = np.sqrt(mse)
acc1 = regressor.score(x_test,y_test)*100
mae = mean_absolute_error(y_test,y_predicted_d)
rmse = np.sqrt(mse)
print("Cay quyet dinh")
print("Mean Squared Error:",mse)
print("Root Mean Squared Error:", rmse)
print("Do chinh xac:",acc1)
print("Mean Absolute Error:",mae)
print ("Thoi gian thuc thi: %.4f (s)" % (end_time - start_time))
print("\n")
plt.title("Cây quyết định")
plt.scatter(y_test,y_predicted_d)
plt.xlabel("Test")
plt.ylabel("Predict")
plt.grid(True)
plt.show()

from prettytable import PrettyTable
table = PrettyTable()
table.field_names = ["Giai Thuat", "MSE","RMSE" ,"MAE","Do chinh xac"]

models = [
    LinearRegression(),
    DecisionTreeRegressor(random_state = 100)  
]

for model in models:
    model.fit(x_train, y_train) 
    y_res = model.predict(x_test)
    mse = mean_squared_error(y_test, y_res)
    score = model.score(x_test, y_test)*100  
    mae = mean_absolute_error(y_test,y_res)
    rmse = np.sqrt(mse)
    table.add_row([type(model).__name__, format(mse, '.2f'),format(rmse, '.2f'),format(mae, '.2f'),format(score, '.2f')])
plt.show()
print(table)