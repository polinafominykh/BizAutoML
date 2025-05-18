from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
import numpy as np


def train_model(model, X, y, task_type: str, test_size=0.2, random_state=42):
    """
    Обучает модель и возвращает обученную модель и метрику

    :param model: sklearn-модель
    :param X: DataFrame — признаки
    :param y: Series — целевая переменная
    :param task_type: str — 'regression' или 'classification'
    :return: обученная модель, значение метрики
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task_type == 'regression':
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        metric = f"MAE: {mae:.2f}, RMSE: {rmse:.2f}"
    elif task_type == 'classification':
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        metric = f"Accuracy: {acc:.2f}, F1-score: {f1:.2f}"
    else:
        raise ValueError(f"Неизвестный тип задачи: {task_type}")

    return model, metric
