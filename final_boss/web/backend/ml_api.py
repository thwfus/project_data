from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import pandas as pd

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
pipe = joblib.load(BASE_DIR / "logreg_pipeline.pkl")

# 21 cột theo đúng thứ tự in ra
FEATURES = [
    'HighBP','HighChol','CholCheck','BMI','Smoker','Stroke',
    'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies',
    'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth',
    'MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income'
]

class Input(BaseModel):
    Sex: int | None = None
    Age: int | None = None
    BMI: float | None = None
    HighBP: int | None = None
    HighChol: int | None = None
    Smoker: int | None = None
    PhysActivity: int | None = None
    Fruits: int | None = None
    Veggies: int | None = None
    MentHlth: int | None = None
    GenHlth: int | None = None
    PhysHlth: int | None = None
    Education: int | None = None
    Income: int | None = None
    Stroke: int | None = None
    HeartDiseaseorAttack: int | None = None
    CholCheck: int | None = None
    AnyHealthcare: int | None = None
    NoDocbcCost: int | None = None
    DiffWalk: int | None = None
    HvyAlcoholConsump: int | None = None


@app.post("/predict")
def predict(inp: Input):
    # mặc định toàn bộ = 0
    row = {f: 0 for f in FEATURES}

    # overwrite các field user nhập
    data = inp.model_dump(exclude_none=True)
    for k, v in data.items():
        row[k] = v

    X = pd.DataFrame([row], columns=FEATURES)

    # LogisticRegression → lấy xác suất class 1
    proba = float(pipe.predict_proba(X)[0][1])

    return {
        "ai_probability": proba,
        "ai_percent": round(proba * 100, 2),
        "used_features": FEATURES,
        "input_used": row
    }
