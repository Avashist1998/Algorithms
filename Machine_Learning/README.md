# Machine Learning 

## Decision_Stamp 

>The base of the alogrithm is sourced from the Book "Theory to Algorthim". In my implemnetation, the algorithm is designed to perform efficently over larger feature space.

### Sample Code

```
from Machine_Learning.Decision_Stramp import Decision_Stamp
my_Stamp = Decision_Stamp()
X_train, X_test, y_train, y_test = my_train_test_split(
    X, y, test_size=0.20, random_state=42)
my_Stamp.fit(X_train, y_train)
prediction = my_Stamp.predict(X_test)
score = my_Stamp.accuracy(prediction,y_test)
```

