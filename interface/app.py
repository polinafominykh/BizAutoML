import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

from core.preprocessing import preprocess_data
from core.task_detector import detect_task_type
from core.model_selector import select_model
from core.trainer import train_model
from core.visualizer import plot_regression, plot_classification, plot_feature_importance
from core.exporter import (
    save_model,
    save_predictions,
    save_metrics,
    save_all_as_zip,
    generate_pdf_report
)

st.set_page_config(page_title="BizAutoML", layout="wide")
st.title("📊 BizAutoML — Автоматический ML-анализ для малого бизнеса")

file = st.file_uploader("🗂️ Загрузите CSV-файл", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("👁️‍🗨️ Предпросмотр данных")
    st.dataframe(df.head())

    target = st.selectbox("🎯 Выберите целевую переменную", df.columns)

    if st.button("🚀 Запустить анализ"):
        with st.spinner("Обработка данных и обучение модели..."):
            # Обработка и определение задачи
            X, y, scaler, encoders = preprocess_data(df, target)
            task = detect_task_type(y)
            model = select_model(task)
            model, metric = train_model(model, X, y, task)
            y_pred = model.predict(X)

        st.success(f"✅ Тип задачи: {task}")
        st.info(f"📈 Метрика модели: {metric}")

        st.subheader("📉 Визуализация")
        if task == "regression":
            plot_regression(y, y_pred)
        elif task == "classification":
            plot_classification(y, y_pred)

        plot_feature_importance(model, X.columns)

        # Сохранение
        save_model(model)
        save_predictions(y, y_pred)
        save_metrics(metric)

        # PDF отчёт
        pred_df = pd.DataFrame({"Actual": y, "Predicted": y_pred})
        pdf_path = generate_pdf_report(task, metric, df.head(), pred_df)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📝 Скачать PDF-отчёт",
                data=f,
                file_name="BizAutoML_report.pdf",
                mime="application/pdf"
            )

        # ZIP архив
        zip_path = save_all_as_zip()
        with open(zip_path, "rb") as f:
            st.download_button(
                label="🗂️ Скачать всё в ZIP",
                data=f,
                file_name="BizAutoML_results.zip",
                mime="application/zip"
            )

        st.subheader("💾 Экспорт")
        st.markdown("- 🔐 Модель: `output/model.joblib`")
        st.markdown("- 📄 Предсказания: `output/predictions.csv`")
        st.markdown("- 📊 Метрики: `output/metrics.json`")
        st.markdown("- 📝 Отчёт: `output/report.pdf`")
