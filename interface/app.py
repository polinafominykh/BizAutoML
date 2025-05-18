import streamlit as st
import pandas as pd

from core.preprocessing import preprocess
from core.task_detector import detect_task_type
from core.model_selector import select_model
from core.trainer import train_model

st.title("BizAutoML — анализ данных для бизнеса")

file = st.file_uploader("Загрузите CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("Предпросмотр данных:", df.head())

    target = st.selectbox("Выберите целевую переменную", df.columns)

    if st.button("Запустить анализ"):
        X = df.drop(columns=[target])
        y = df[target]

        X_proc = preprocess(X)
        task = detect_task_type(y)
        model = select_model(task)
        model, metric = train_model(model, X_proc, y, task)

        st.success(f"Тип задачи: {task}")
        st.info(f"Метрика модели: {metric}")
