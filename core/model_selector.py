from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def select_model(task_type: str):
    """
    Выбирает модель на основе типа задачи

    :param task_type: str — 'classification' или 'regression'
    :return: модель sklearn
    """
    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif task_type == 'regression':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Неизвестный тип задачи: {task_type}")

    return model
