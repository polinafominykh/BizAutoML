def detect_task_type(y, threshold=10):
    """
    Определяет тип задачи: классификация или регрессия
    :param y: pandas Series — целевая переменная
    :param threshold: сколько уникальных значений считается максимумом для классификации
    :return: str — 'classification' или 'regression'
    """
    if y.dtype == 'object':
        return 'classification'

    unique_values = y.nunique()

    if y.dtype.kind in 'biu' and unique_values <= threshold:
        return 'classification'

    return 'regression'
