import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
# df['BMI_Age'] = df['BMI'] * df['Age']
# df['Income_Edu'] = df['Income'] * df['Education']
# df['PhysBMI'] = df['PhysActivity'] * df['BMI']
#thong ke so tan suat cua tung cot

# for col in df.columns:
#     print(f"\n--- {col} ---")
#     print(df[col].value_counts())
int_cols = [
    'Diabetes_binary', 'HighBP', 'HighChol', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk',
    'Sex', 'GenHlth', 'Education', 'Income'
]

df[int_cols] = df[int_cols].astype('int')

def chuan_hoa_bien():
    #MenHlth: 0–14 => 0 (good), 15–30 => 1 (bad)
    df["MentHlth_bin"] = df["MentHlth"].apply(lambda x: 0 if pd.notna(x) and 0 <= x <= 14 else 1)
    # PhysHlth: 0–14 => 0 (good), 15–30 => 1 (bad)
    df["PhysHlth_bin"] = df["PhysHlth"].apply(lambda x: 0 if pd.notna(x) and 0 <= x <= 14 else 1)
    # BMI: 1 neu “ổn” (18.5–24.9), con lai la 0 (thap/cao)
    df["BMI_bin"] = df["BMI"].apply(lambda x: 1 if pd.notna(x) and 18.5 <= x <= 24.9 else 0)
    df.drop(columns=["MentHlth", "PhysHlth", "BMI"], inplace=True)


def phan_bo_bien_muc_tieu():
    # Bar chart cho bien muc tieu
    plt.figure(figsize=(6,4))
    df["Diabetes_binary"].value_counts().plot(kind="bar", color=["skyblue","salmon"])
    plt.title("Phân bố biến mục tiêu: Diabetes_binary")
    plt.xlabel("Diabetes (0 = Không, 1 = Có)")
    plt.ylabel("Số lượng")
    plt.show()

# Heatmap ma tran tuong quan
def heatmap_ma_tran_tuong_quan():
    plt.figure(figsize=(12,8))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("ma trận tương quan giữa các biến")
    plt.show()


def loc_bien_tuong_quan():
    global df
    target = "Diabetes_binary"
    # tinh ma tran tuong quan pearson
    corr_matrix = df.corr()
    # lay tuong quan voi bien muc tieu
    corr_with_target = corr_matrix[target].drop(target)
    # loc cac bien co gia tri tuong quan >= threshold
    threshold = 0.09
    selected_features = corr_with_target[abs(corr_with_target) >= threshold].index.tolist()
    print("Các biến giữ lại")
    print(selected_features)
    print("Tương quan với Diabetes_binary ")
    print(corr_with_target[abs(corr_with_target) >= threshold])
    df = df[selected_features + [target]]
    
def kiem_tra_da_cong_tuyen_VIF():
    global df
    X = add_constant(df.drop("Diabetes_binary", axis=1))
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data)
    #loc bien co VIF > 5
    features_to_keep = vif_data[vif_data["VIF"] <= 5]["feature"].tolist()
    features_to_keep.append("Diabetes_binary")  # them lai bien muc tieu
    df = df[features_to_keep]

#chuan_hoa_bien()
#loc_bien_tuong_quan()
#phan_bo_bien_muc_tieu()
#heatmap_ma_tran_tuong_quan()

#kiem_tra_da_cong_tuyen_VIF()

# print(df.head())
# print(df.info())
