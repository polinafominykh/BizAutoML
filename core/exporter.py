import joblib
import pandas as pd
import json
import os
import zipfile
from datetime import datetime

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_model(model, filename="model.joblib"):
    path = os.path.join(OUTPUT_DIR, filename)
    joblib.dump(model, path)
    return path

def save_predictions(y_true, y_pred, filename="predictions.csv"):
    df = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred
    })
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    return path

def save_metrics(metrics_str, filename="metrics.json"):
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics_str
    }
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    return path

def save_all_as_zip(zip_filename="BizAutoML_results.zip"):
    zip_path = os.path.join(OUTPUT_DIR, zip_filename)
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for filename in ["model.joblib", "predictions.csv", "metrics.json"]:
            file_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.exists(file_path):
                zipf.write(file_path, arcname=filename)
    return zip_path
from fpdf import FPDF

from fpdf import FPDF
import os
from datetime import datetime

def sanitize(text):
    """Удаляет символы, не поддерживаемые latin-1"""
    try:
        return str(text).encode('latin-1', 'ignore').decode('latin-1')
    except Exception:
        return str(text)

def generate_pdf_report(task_type, metrics_str, df_preview, predictions_df, filename="report.pdf"):
    path = os.path.join("output", filename)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="BizAutoML - Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt=f"Task type: {sanitize(task_type)}", ln=True)
    pdf.cell(200, 10, txt=f"Metrics: {sanitize(metrics_str)}", ln=True)
    pdf.ln(10)

    pdf.cell(200, 10, txt="Example input rows:", ln=True)
    for _, row in df_preview.iterrows():
        pdf.multi_cell(0, 10, sanitize(row.to_dict()))

    pdf.ln(5)
    pdf.cell(200, 10, txt="Example predictions:", ln=True)
    for _, row in predictions_df.head().iterrows():
        pdf.cell(200, 10, txt=f"Actual: {row['Actual']} | Predicted: {row['Predicted']}", ln=True)

    pdf.output(path)
    return path
