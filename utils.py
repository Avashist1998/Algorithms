import numpy as np

def encoder (y):
    unique = list(np.unique(y))
    encoder = lambda x: unique.index(x)
    y_new = np.array(list(map(encoder, y)))
    y_new = 2*y_new - 1
    return y_new


def my_train_test_split(X, y, test_size=0.2, random_state=42):
    [row, _] = X.shape
    index = np.arange(row)
    np.random.seed(random_state)
    np.random.shuffle(index)
    stop = int((row-1)*(1-test_size))
    X_train, y_train = X[index[:stop]], y[index[:stop]]
    X_test, y_test = X[index[stop:]], y[index[stop:]]
    return X_train, X_test, y_train, y_test