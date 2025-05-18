from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def select_model(task):
    if task == 'classification':
        return RandomForestClassifier()
    else:
        return RandomForestRegressor()
