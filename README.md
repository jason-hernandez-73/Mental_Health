# Machine Learning - How does Unemployment related to Mental illness?

This project tries to examine whether there are any predicted variables that could impact mental illness such as income. We first use the numeric matrix to pick out the more related variables on mental illess, then we build machines learning models like decision tree and random foreset to find out the most important variable. 

## Getting Started

First, we download the Unemployment and Mental Illness survey Dataset https://www.kaggle.com/michaelacorley/unemployment-and-mental-illness-survey

Second, we cleaned the table and transform them into right format through using Pandas.

### Installing 
Things that you need to install in your terminal 
Basic cleaning
```
pip install pandas
pip install numpy
```

### Select the most possible variable
Preview with numeric matrix and chart like heatmap
```
pip install matplotlib
pip install seaborn
cleaned_dataframe.corr()
```
```
pearsoncorr=cleaned_dataframe.corr(method='pearson')
import seaborn as sb
sb.heatmap(pearsoncorr,xticklabels=pearsoncorr.columns, yticklabels=pearsoncorr.columns,cmap='RdBu_r', annot=False,linewidth=1)
```
### Test out with a machines learning model 
Install machine learning packages
```
pip install sklearn
pip install tensorflow
```
Preset code to fit more model later
```
from sklearn import metrics
def train_model(classifier, feature_vector_train, train_label, feature_vector_valid, valid_label, is_neural_net=False):
    classifier.fit(feature_vector_train, train_label)
    predictions=classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions=predictions.argmax(axis=-1)
    return metrics.accuracy_score(predictions, valid_label)

```
```
from sklearn.ensemble import RandomForestClassifier
forest_accuracy=train_model(RandomForestClassifier(n_estimators=50),X_train_scaled,y_train, X_test_scaled,y_test)
print(forest_accuracy)

```
```
rf=RandomForestClassifier().fit(X_train_scaled,y_train)
r=rf.score(X_test_scaled,y_test)
print(r)
```
```
sorted(zip(rf.feature_importances_,xd_df),reverse=True)
```

### Conclusion

When we look at household income as the predicted variable, household income is not very predictable; even narrowing it down to just the most relevant columns still resulted in little predictive power.The only highly correlated data relative to factors likely to affect income was the obvious: mental illness, disability, and welfare all predicted against being employed!

## Built With

* [Jupyter Notebook](https://jupyter.org/) - usually initiated from computer's terminal
* [Tableau](https://public.tableau.com/en-us/s/) - More visualized graph and chart 


## Authors
* [Jason Hernandez] (https://github.com/jason-hernandez-73)
* [Weiqi Liang (Vicky)] (https://github.com/liangweiqi2)

## Acknowledgments

* Our instructor Kevin Lee from University of California Berkeley Extension Data Visualization Bootcamp

