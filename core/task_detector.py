def detect_task_type(y):
    if y.nunique() <= 10 or y.dtype == 'object':
        return 'classification'
    return 'regression'
