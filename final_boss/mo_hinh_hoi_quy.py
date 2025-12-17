from utils.lib import *
import file3_binary_da
from sklearn.pipeline import Pipeline
import joblib

#df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
y=file3_binary_da.df["Diabetes_binary"]
X=file3_binary_da.df.drop("Diabetes_binary",axis=1)
#chia du lieu thanh tap train va test
#co cai web de coi tham so:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=7749))
])

pipe.fit(X_train, y_train)

joblib.dump(pipe, "logreg_pipeline.pkl")
y_pred = pipe.predict(X_test)

def the_result():
    
    # do chinh xac
    print("Accuracy:", accuracy_score(y_test, y_pred))
    # ma tran nham lan
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    # bao cao chi tỉet
    print("Classification Report:\n", classification_report(y_test, y_pred))
    

print("SO COT:", X.shape[1])
print("DANH SACH COT:")
print(X.columns.tolist())
for i, c in enumerate(X.columns):
    print(i, c)
