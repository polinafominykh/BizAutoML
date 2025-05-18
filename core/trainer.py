from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

def train_model(model, X, y, task):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task == 'classification':
        metric = accuracy_score(y_test, y_pred)
    else:
        metric = mean_absolute_error(y_test, y_pred)

    return model, metric
