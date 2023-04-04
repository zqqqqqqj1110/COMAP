import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel('双体船.xlsx')
X = data.iloc[:, 1:7].values
y = data.iloc[:, 0].values


# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)


# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# print(X_train)

# 构建模型
model = Sequential()
model.add(Dense(units=50, activation='sigmoid', input_dim=6))
model.add(Dense(units=50, activation='sigmoid'))
model.add(Dense(units=50, activation='sigmoid'))
model.add(Dense(units=1))

# 编译模型
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test))

# 使用测试集进行预测
y_pred = model.predict(X_test)

# print(y_pred)

# 可视化结果
plt.scatter(y_test, y_pred)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values')
plt.show()

# 损失函数
history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# 预测误差分布图
error = y_test - y_pred
plt.hist(error, bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Prediction Error Distribution')
plt.show()

