# Machine Learning 
>The base of the alogrithm is sourced from the Book "Theory to Algorthim"

## Decision_Stamp 


#### Sample Code
```
    from machine_learning.decision_stamp import decision_stamp
    my_stamp = decision_stamp()
    my_stamp.fit(X_train, y_train)
    prediction = my_stamp.predict(X_test)
    score = my_stamp.accuracy(prediction,y_test)
```

## Adaboost 

### Sample Code 
```
    from machine_learning.adaboost import adaboost
    my_ada = adaboost()
    my_ada.fit(X_train, y_train)
    prediction = my_ada.predict(X_test)
    score = my_ada.accuracy(prediction,y_test)
```