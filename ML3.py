import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
dta = sm.datasets.fair.load_pandas().data

# add "affair" column: 1 represents having affairs, 0 represents not
dta['affair'] = (dta.affairs > 0).astype(int)
#Data expoloration 
dta.groupby('affair').mean()
y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + \
religious + educ + C(occupation) + C(occupation_husb)',
dta, return_type="dataframe")

X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
'C(occupation)[T.3.0]':'occ_3',
'C(occupation)[T.4.0]':'occ_4',
'C(occupation)[T.5.0]':'occ_5',
'C(occupation)[T.6.0]':'occ_6',
'C(occupation_husb)[T.2.0]':'occ_husb_2',
'C(occupation_husb)[T.3.0]':'occ_husb_3',
'C(occupation_husb)[T.4.0]':'occ_husb_4',
'C(occupation_husb)[T.5.0]':'occ_husb_5',
'C(occupation_husb)[T.6.0]':'occ_husb_6'})
y = np.ravel(y)

X.hist()
# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, y)
# check the accuracy on the training set
print("\nModel score: ",model.score(X, y))
print("\n73% accuracy seems good, but what's the null error rate?")

print("\n what percentage had affairs?")
print(y.mean())
print("\nOnly 32% of the women had affairs, which means that you could obtain 68% accuracy by always predicting 'no'. So we're doing better than the null error rate, but not by much")
C1=np.transpose(model.coef_)
C=X.columns
pd.DataFrame(C1,index=C ,columns={"Model Coeff.."})

print("\n\nModel Evaluation Using a Validation Set")
# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)
print("\nWe now need to predict class labels for the test set. We will also generate the class probabilities,")
# predict class labels for the test set
predicted = model2.predict(X_test)
print("\n",predicted)

# generate class probabilities
probs = model2.predict_proba(X_test)
print("\n",probs)

print("""\nAs you can see, the classifier is predicting a 1 (having an affair) any time the probability in the second column is greater than 0.5.

Now let's generate some evaluation metrics.""")

# generate evaluation metrics
print(metrics.accuracy_score(y_test, predicted))
print( metrics.roc_auc_score(y_test, predicted))

print("""\nThe accuracy is 73%, which is the same as we experienced when training and predicting on the same data.

We can also see the confusion matrix and a classification report with other metrics.""")

print( metrics.confusion_matrix(y_test, predicted))
print( metrics.classification_report(y_test, predicted))

print("\n\nModel Evaluation Using Cross-Validation")
print("\nNow let's try 10-fold cross-validation, to see if the accuracy holds up more rigorously.")

# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print( scores)
print( scores.mean())

print("\nLooks good. It's still performing at 73% accuracy.")

