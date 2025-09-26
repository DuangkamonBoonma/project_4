import io
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import LabelEncoder
import json
from pathlib import Path

# ---------------- App ----------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------- Load Model ----------------
model = joblib.load("best_model_xgb.pkl")

# โหลดค่าคาลิเบรชัน (ถ้ามี)
calib_path = Path("calib.json")
if calib_path.exists():
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    Q01, Q99, T0, S = calib["q01"], calib["q99"], calib["t0"], calib["s"]
else:
    Q01, Q99, T0, S = 10.0, 55.0, 35.0, 5.0

def dbz_to_percent(dbz: float) -> float:
    """map dBZ -> [0,100] ด้วย logistic + robust clipping"""
    x = np.clip(dbz, Q01, Q99)
    z = (x - T0) / max(S, 1e-6)
    p = 100.0 * (1.0 / (1.0 + np.exp(-z)))
    return float(np.clip(p, 0.0, 100.0))

# 👉 ฟีเจอร์ที่โมเดลเก่าใช้จริง
final_features = [
    "MML q", "จังหวัด", "K", "Lifted", "Day", "อำเภอ",
    "LCL P.1", "LCL Temp", "SWEAT", "LCL P",
    "Month", "Showalter", "Cross", "Year"
]

# ---------------- Routes ----------------
@app.get("/", response_class=HTMLResponse)
async def page1(request: Request):
    return templates.TemplateResponse("page1.html", {"request": request})

@app.get("/page2.html", response_class=HTMLResponse)
async def page2(request: Request):
    return templates.TemplateResponse("page2.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # --- อ่านไฟล์ ---
        content = await file.read()
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            return JSONResponse({"error": "❌ รองรับเฉพาะไฟล์ .csv หรือ .xlsx"}, status_code=400)

        df.columns = df.columns.astype(str).str.strip()

        # --- ตรวจข้อมูล ---
        if len(df) < 5:
            return JSONResponse({"error": "❌ ต้องมีข้อมูลย้อนหลังอย่างน้อย 5 วัน"}, status_code=400)

        # --- เวลา ---
        if "Date" in df.columns:
            df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
            df["Year"]  = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month
            df["Day"]   = df["Date"].dt.day

        # เก็บจังหวัด/อำเภอจากแถวล่าสุด
        latest = df.iloc[-1]
        province_txt = str(latest.get("จังหวัด", "-")) if pd.notna(latest.get("จังหวัด", np.nan)) else "-"
        district_txt = str(latest.get("อำเภอ", "-"))  if pd.notna(latest.get("อำเภอ", np.nan))  else "-"

        # --- เลือกเฉพาะฟีเจอร์ที่โมเดลใช้ ---
        df = df[[c for c in final_features if c in df.columns]].dropna(how="any")
        df = df[final_features]   # ✅ บังคับเรียงลำดับตรงกับโมเดลเก่า

        # --- encode จังหวัด/อำเภอ ---
        le_province, le_district = LabelEncoder(), LabelEncoder()
        if "จังหวัด" in df.columns:
            df["จังหวัด"] = le_province.fit_transform(df["จังหวัด"].astype(str))
        if "อำเภอ" in df.columns:
            df["อำเภอ"] = le_district.fit_transform(df["อำเภอ"].astype(str))

        # ใช้ข้อมูล 5 แถวล่าสุด
        last5 = df.tail(5)

        # ===== พยากรณ์วันนี้ + 3 วัน =====
        preds = []
        today = datetime.now()
        for i in range(4):
            dt = today + timedelta(days=i)
            X_input = last5.copy()
            X_input["Year"], X_input["Month"], X_input["Day"] = dt.year, dt.month, dt.day

            prob_dbz = float(model.predict(X_input).mean())
            chance   = dbz_to_percent(prob_dbz)

            if chance < 30:
                sev, color = "ต่ำ", "#2ecc71"
            elif chance < 50:
                sev, color = "ปานกลาง", "#f1c40f"
            elif chance < 75:
                sev, color = "สูง", "#ff9f1c"
            else:
                sev, color = "รุนแรงมาก", "#e74c3c"

            preds.append({
                "date": dt.strftime("%d/%m/%Y"),
                "probability": round(chance, 1),
                "severity": sev,
                "color": color,
                "province": province_txt,
                "district": district_txt
            })

        return {"predictions": preds}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
