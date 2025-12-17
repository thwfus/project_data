# python file_1.py

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
#
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
#
from sklearn.datasets import load_diabetes

# # Tải dữ liệu
# diabetes = load_diabetes()

# # Tạo DataFrame từ dữ liệu
# df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# # Thêm cột target (giá trị cần dự đoán)
# df['target'] = diabetes.target

# # Xuất ra file CSV
# df.to_csv('diabetes_dataset.csv', index=False)


#-------------------------------------------------------------------------------------------------#

# # PART 1: XỬ LÝ DATASET

df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
# df = pd.read_excel("dataset_CDC.xlsx")
# df = pd.read_excel("diabetes_python.xlsx")
# binary_mapping = {'No': 0, 'Yes': 1, 'Negative': 0, 'Positive': 1, 'Male': 0, 'Female': 1}

# # Áp dụng mapping cho tất cả các column phân loại
# categorical_columns = ['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 
#                       'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
#                       'Itching', 'Irritability', 'delayed healing', 'partial paresis',
#                       'muscle stiffness', 'Alopecia', 'Obesity', 'class']

# for col in categorical_columns:
#     df[col] = df[col].map(binary_mapping)

# df.to_csv('diabetes_normalized.csv', index=False)
#-------------------------------------------------------------------------------------------------#

# # PART 2: KIỂM TRA CAC BIẾN ĐẶC BIỆT LIÊN QUAN ĐẾN diabetes

# Tính hệ số tương quan
# df = pd.read_csv("diabetes_normalized.csv")
# corr = df.corr()

# # Vẽ heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
#             cbar=True, annot_kws={"size":6})  # giảm cỡ số trong ô

# plt.xticks(rotation=45, ha='right', fontsize=7)  # giảm chữ trục X
# plt.yticks(fontsize=7)  

# plt.title("Bảng hệ số tương quan (Correlation Matrix)", fontsize=14)
# plt.tight_layout()
# plt.show()

# Vẽ boxplot cho biến
# sns.boxplot(x=df['target'])
# plt.title("Boxplot of Target")
# plt.show()

# sns.boxplot(x=df['bp'])
# plt.title("Boxplot of Blood Pressure")
# plt.show()
#-------------------------------------------------------------------------------------------------#

# # PART 3: KIỂM TRA ĐA CỘNG TUYẾN HOÀN HẢO
# cols = ["HighBP", "HighChol", "BMI", "HeartDiseaseorAttack", "GenHlth", "PhysHlth", "DiffWalk", "Age", "PhysActivity", "Education", "Income"]
# X = add_constant(df[cols])

# vif = pd.DataFrame({
#     "Variable": X.columns,
#     "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# })

# print(vif)

#-------------------------------------------------------------------------------------------------#
# # TRAIN MODEL
# Xác định biến phụ thuộc (y) và độc lập (X)
X = df[["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]]  # Giả sử tất cả các cột trừ cột cuối cùng là biến độc lập
y = df["Diabetes_012"]

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# # PART 4: MÔ HÌNH HỒI QUY LOGISTIC
# # Huấn luyện mô hình logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá mô hình
print("Độ chính xác (Accuracy):", accuracy_score(y_test, y_pred))
print("\nBáo cáo phân loại (Classification Report):\n", classification_report(y_test, y_pred))

# Ma trận nhầm lẫn
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.show()

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # Huấn luyện mô hình Linear Regression
# model = LinearRegression()
# model.fit(X_train, y_train)

# import numpy as np
# # Dự đoán
# y_pred = model.predict(X_test)

# # Đánh giá mô hình
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# print("MSE:", mse)
# print("RMSE:", np.sqrt(mse))
# print("R² score:", r2_score(y_test, y_pred))

# # Vẽ biểu đồ dự đoán vs thực tế
# plt.scatter(y_test, y_pred)
# plt.xlabel("Giá trị thực tế")
# plt.ylabel("Giá trị dự đoán")
# plt.title("Actual vs Predicted")
# plt.show()

#-------------------------------------------------------------------------------------------------#
# # PART 5: MÔ HÌNH CÂY QUYẾT ĐỊNH VÀ KNN
# model_tree = DecisionTreeClassifier(random_state=42)
# model_tree.fit(X_train, y_train)
# y_pred_tree = model_tree.predict(X_test)

# print(confusion_matrix(y_test, y_pred_tree))
# print(classification_report(y_test, y_pred_tree))

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import mean_squared_error, r2_score

# # Generate synthetic dataset
# X = df.iloc[:, :-1].values  # 10 biến đầu tiên
# y = df.iloc[:, -1].values # Biến mục tiêu (cuối cùng)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the KNN regressor
# knn_regressor = KNeighborsRegressor(n_neighbors=5)
# knn_regressor.fit(X_train, y_train)

# # Make predictions on the test data
# y_pred = knn_regressor.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error: {mse}')
# print(f'R-squared: {r2}')

# # Visualize the results
# plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
# plt.xlabel('Actual Target')
# plt.ylabel('Predicted Target')
# plt.title('KNN Regression')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # đường y=x
# plt.legend()
# plt.show()




# CHỖ LẤY DATASET
# dummyjson
# SerpApi
# jupyter notebook
# data gov


