import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

def plot_regression(y_test, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Фактические значения")
    ax.set_ylabel("Прогноз")
    ax.set_title("Прогноз vs Факт")
    st.pyplot(fig)

def plot_classification(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Факт")
    ax.set_title("Матрица ошибок")
    st.pyplot(fig)

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        data = pd.Series(importances, index=feature_names).sort_values(ascending=True)
        fig, ax = plt.subplots()
        data.plot(kind='barh', ax=ax)
        ax.set_title("Важность признаков")
        st.pyplot(fig)
