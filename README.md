This package gives you the opportunity to use a Target mean Encoding.

TargetEncoder - The algorithm encodes all features that are submitted to the input based on the target.

Parameters:

    alpha: float or int, smoothing for generalization.
![fifth](img/5.png)

    max_unique: int, maximum number of unique values in a feature. 
                If there are more unique values inside the feature,
                then the algorithm will split this feature into bins, 
                the number of max_unique.
                
    split: list of int or cross-validator class,
                if split is [], then algorithm will encode features without cross-validation
                This situation features will overfit on target

                if split len is 1 for example [5], algorithm will encode features by using cross-validation on 5 folds
                This situation you will not overfit on tests, but when you will validate, your score will overfit

                if split len is 2 for example [5, 3], algorithm will separate data on 5 folds, afterwords
                will encode features by using cross-validation on 3 folds
                This situation is the best way to avoid overfit, but algorithm will use small data for encode.
---
TargetEncoderRegressor - The algorithm encodes all feature and then takes the average of encoded features as prediction.

    alpha: float or int, smoothing for generalization.

    max_unique: int, maximum number of unique values in a feature. 
                If there are more unique values inside the feature,
                then the algorithm will split this feature into bins, 
                the number of max_unique.
    
    used_features: int, this is a number of used features for prediction
                   The algorithm encodes all features with the average value of the target, 
                   then the std is considered inside each feature,
                   and "used_features" features with the highest std are selected to use only informative features. 
---
TargetEncoderClassifier - The algorithm encodes all feature and then takes the average of encoded features as prediction.

    alpha: float or int, smoothing for generalization.

    max_unique: int, maximum number of unique values in a feature. 
                If there are more unique values inside the feature,
                then the algorithm will split this feature into bins, 
                the number of max_unique.
    
    used_features: int, this is a number of used features for prediction
                   The algorithm encodes all features with the average value of the target, 
                   then the std is considered inside each feature,
                   and "used_features" features with the highest std are selected to use only informative features. 
             
---
Categorical features can be encoded in several ways. The first method is to encode just numbers from 0 to n-1, where n is the number of unique values. Such an encoding is called LabelEncoding.

![first](img/1.png)

Here we coded
"Moscow": 0,
"New York": 1,
"Rome": 2

Another encoding method is called OneHotEncoding. Here we create instead of a single feature n features, where n is the number of unique values. Where for each object we put 0 everywhere except for the k-th element, where there is 1.

![second](img/2.png)

Another method of encoding categorical features is used here - encoding by the average value of the target.

![third](img/3.png)

Average encoding is better than LabelEncoding, because a histogram of predictions using label & mean encoding show that mean encoding tend to group the classes together whereas the grouping is random in case of LabelEncoding.
![fourth](img/4.png)

___

Consider next example, here is a table with information about the categories in the data. It can be seen that there are several categories, the number of which is very small, or did not occur in the dataset. Such data can interfere with the model, and this data can be retrained. As you can see Rome was presented only once and its target was 0, then whenever we encode Rome we will replace it with 0. And that's the problem, our algorithm will be retrained. To avoid this, we will use smoothing.

![fifth](img/5.png)

![sixth](img/6.png)

As you can see, we were able to solve the problem with small classes, their encodings have become more smoothed and shifted to the mean values.

___

Next we will be able to encode Train dataset and Test dataset.

In order to avoid overfitting, we have to use the Folds split when encoding on the Train, and if we use validation that would on validation also not to retrain we inside each Fold have to do another split on the Folds.
And for Test dataset, we use all the data from Train dataset for encoding.

![seventh](img/7.png)

___

Once we have coded average, there are 3 uses for these features. 
1. Train the model on our new data.
2. Train the model on our new and old data.
3. Take the average of the new data and use it as a prediction.

In the folder "experiments" have the results from the comparison of these methods.

___

Example of usage
```python
from target_encoding import TargetEncoderClassifier
from target_encoding import TargetEncoder

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

enc = TargetEncoder()
new_X_train = enc.transform_train(X=X_train, y=y_train)
new_X_test = enc.transform_test(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict_proba(X_test)[:,1]
print('without target encoding', roc_auc_score(y_test, pred))

rf.fit(new_X_train, y_train)
pred = rf.predict_proba(new_X_test)[:,1]
print('with target encoding', roc_auc_score(y_test, pred))

enc = TargetEncoderClassifier()
enc.fit(X_train, y_train)
pred = enc.predict_proba(X_test)[:,1]
print('target encoding classifier', roc_auc_score(y_test, pred))
```
```
without target encoding 0.9952505732066819
with target encoding 0.996560759908287
target encoding classifier 0.9973796265967901

```

___
You can install it by using pip
```
pip install target_encoding
```

___
```
Requirements:
    numpy==1.16.2
    scikit-learn==0.20.3
```