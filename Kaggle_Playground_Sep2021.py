#import necessary modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#read in data
train_data = pd.read_csv("C:/Users/ankur/PycharmProjects/KaggleCompetitions/Playground_Sep2021/train.csv")
test_data = pd.read_csv("C:/Users/ankur/PycharmProjects/KaggleCompetitions/Playground_Sep2021/test.csv")

#get a first look at the data
print(train_data.head())

#define our X and y as well as X_test
X = train_data.drop(["id","claim"], axis = 1)
y = train_data["claim"]
X_test = test_data.drop(["id"],axis = 1)
#the output of y.value_counts() shows us that we're dealing with binary classification
print(y.value_counts())

#let's plot our features
sns.set()
curr_plot = 1
plt.figure()
fig,ax = plt.subplots(24,5,figsize = (28,75))
for row in range(24):
    for col in range(5):
        if curr_plot < 119:
            sns.histplot(data = X.iloc[:,curr_plot],kde=True,ax=ax[row,col]).set(ylabel="")
            curr_plot += 1
#plt.savefig("C:/Users/ankur/PycharmProjects/KaggleCompetitions/Playground_Sep2021/EDA_histogram_train.png")

curr_plot = 1
plt.figure()
fig,ax = plt.subplots(24,5,figsize = (28,75))
for row in range(24):
    for col in range(5):
        if curr_plot < 119:
            sns.histplot(data = X_test.iloc[:,curr_plot],kde=True,color="green",ax=ax[row,col]).set(ylabel="")
            curr_plot += 1
#plt.savefig("C:/Users/ankur/PycharmProjects/KaggleCompetitions/Playground_Sep2021/EDA_histogram_test.png")


missing_byfeature = X.isna().sum()
max_missing_features = np.max(missing_byfeature)
min_missing_features = np.min(missing_byfeature)

#now we need to do some scaling and imputation
SS = StandardScaler()
#we need to make sure we preserve the column names and keep X as a DataFrame object
X = pd.DataFrame(SS.fit_transform(X))
X.columns = list(train_data.iloc[:,1:119].columns)
print(X.head())
#We'll scale and impute using the median
SI_median = SimpleImputer(missing_values=np.nan,strategy="median")
X = pd.DataFrame(SI_median.fit_transform(X))
X.columns = list(train_data.iloc[:,1:119].columns)

#We need to do scaling and imputation for the test data as well
X_test = pd.DataFrame(SS.fit_transform(X_test))
X_test.columns = list(test_data.iloc[:,1:].columns)
X_test = pd.DataFrame(SI_median.fit_transform(X_test))
X_test.columns = list(test_data.iloc[:,1:].columns)

#Make our training and validation sets
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.3,random_state=42)

#we need to write our own roc_auc_score function
def my_roc_auc_score(model,X,y):
    return roc_auc_score(y,model.predict_proba(X)[:,1])

#let's first try logistic regression with an L1 penalty (regularization similar to LASSO). We use the "saga" solver since
#the dataset is almost 1 million rows.
LogReg_L1 = LogisticRegression(penalty="l1",solver = "saga")
LogReg_L1_cv = cross_val_score(LogReg_L1,X,y,cv=5,scoring=my_roc_auc_score,n_jobs=-1)
auc_LogReg_L1 = np.mean(LogReg_L1_cv)

#Now let's try elastic net with a grid of l1 ratio's
ratios = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
params_dict = {"l1_ratio":ratios}
LogReg_ElasticNet_hp = GridSearchCV(estimator = LogisticRegression(penalty = "elasticnet",solver="saga")
                                    , param_grid=params_dict,cv=5,scoring = my_roc_auc_score)
LogReg_ElasticNet_hp.fit(X,y)
LogReg_ElasticNet_results = pd.DataFrame(LogReg_ElasticNet_hp.cv_results_)
LogReg_ElasticNet_results = LogReg_ElasticNet_results[["rank_test_score","mean_test_score","param_l1_ratio"]]
LogReg_ElasticNet_results.sort_values(by = "rank_test_score",inplace = True)
#using GridSearchCV, we can see that the mean_test_score is virtually identical regardless of the l1_ratio. This means
#elastic net and L2 do not improve over L1.
print(LogReg_ElasticNet_results)
LogReg_ElasticNet_results.to_csv("C:/Users/ankur/PycharmProjects/KaggleCompetitions/Playground_Sep2021/LogisticRegression_CV.csv")

#Logistic regression is not yielding satisfactory results. We will now turn to boosting.
xgb_clf1 = xgb.XGBClassifier(learning_rate=0.1,n_estimators=1000,colsample_bytree=0.25,subsample=0.75,n_jobs=-1,max_depth=3
                             ,eval_metric = "auc",use_label_encoder=False,random_state=42)
xgb_clf1_cv = cross_val_score(xgb_clf1,X,y,cv=5,n_jobs=-1)
auc_xgb_clf1_cv = np.mean(xgb_clf1_cv)
xgb_model1 = xgb_clf1.fit(X,y)
y_pred_valid = xgb_model1.predict(X_valid)
fpr,tpr,thresholds = roc_curve(y_valid,y_pred_valid)
auc_xgb_model1 = auc(fpr,tpr)
y_test = xgb_model1.predict_proba(X_test)[:,1]
id = test_data["id"]
submission = pd.concat([pd.Series(id),pd.Series(y_test)],axis = 1)
submission.columns = ["id","claim"]
submission.set_index("id",inplace=True)
submission.to_csv("C:/Users/ankur/PycharmProjects/KaggleCompetitions/Playground_Sep2021/submission1.csv")

#now let's try a grid search over colsample_bylevel. Let's try decreasing the subsample
#value since 0.75 could lead to overfitting. Let's also try increasing the pool of available features for each tree. Note
#that at each level, the proportion of features available will be colsample_bylevel * colsample_bytree
feature_prop_bylevel = [0.25,0.5,0.75]
params_dict = {"colsample_bylevel":feature_prop_bylevel}
xgb_hp1 = GridSearchCV(estimator=xgb.XGBClassifier(learning_rate=0.1,n_estimators=1000,n_jobs=-1,max_depth=3
,eval_metric = "auc",use_label_encoder=False,random_state=42, subsample=0.25,colsample_bytree=0.5)
, param_grid=params_dict,cv=5,scoring=my_roc_auc_score,verbose=4)
xgb_hp1.fit(X,y)
xgb_results1 = pd.DataFrame(xgb_hp1.cv_results_)
xgb_results1 = xgb_results1[["param_colsample_bylevel","mean_test_score","rank_test_score"]]
xgb_results1.sort_values(by = "rank_test_score",inplace=True)
print(xgb_results1)
xgb_results1.to_csv("C:/Users/ankur/PycharmProjects/KaggleCompetitions/Playground_Sep2021/xgb_results1.csv")

xgb_clf2 = xgb.XGBClassifier(learning_rate=0.1,n_estimators=1000,n_jobs=-1,max_depth=3,colsample_bylevel=0.75
,eval_metric = "auc",use_label_encoder=False,random_state=42,subsample=0.25,colsample_bytree=0.5)
xgb_model2 = xgb_clf2.fit(X,y)
y_test = xgb_model2.predict_proba(X_test)[:,1]
id = test_data["id"]
submission = pd.concat([pd.Series(id),pd.Series(y_test)],axis = 1)
submission.columns = ["id","claim"]
submission.set_index("id",inplace=True)
submission.to_csv("C:/Users/ankur/PycharmProjects/KaggleCompetitions/Playground_Sep2021/submission2.csv")

xgb_clf3 = xgb.XGBClassifier(learning_rate=0.1,n_estimators=1000,n_jobs=-1,max_depth=3,colsample_bylevel=0.75
,eval_metric = "auc",use_label_encoder=False,random_state=42,subsample=0.25,colsample_bytree=0.5,reg_alpha=40)
xgb_model3 = xgb_clf3.fit(X,y)
y_test = xgb_model3.predict_proba(X_test)[:,1]
id = test_data["id"]
submission = pd.concat([pd.Series(id),pd.Series(y_test)],axis = 1)
submission.columns = ["id","claim"]
submission.set_index("id",inplace=True)
submission.to_csv("C:/Users/ankur/PycharmProjects/KaggleCompetitions/Playground_Sep2021/submission3.csv")

alpha = [0,10,20,30,40,50,100]
params_dict = {"reg_alpha":alpha}
xgb_hp2 = GridSearchCV(estimator=xgb.XGBClassifier(learning_rate=0.1,n_estimators=1000,n_jobs=-1,max_depth=3
,eval_metric = "auc",use_label_encoder=False,random_state=42, subsample=0.25,colsample_bytree=0.5,colsample_bylevel=0.75)
, param_grid=params_dict,cv=5,scoring=my_roc_auc_score,verbose=4)
xgb_hp2.fit(X,y,verbose=4)
xgb_results2 = pd.DataFrame(xgb_hp2.cv_results_)
xgb_results2 = xgb_results2[["param_colsample_bylevel","mean_test_score","rank_test_score"]]
xgb_results2.sort_values(by = "rank_test_score",inplace=True)
print(xgb_results2)
xgb_results2.to_csv("C:/Users/ankur/PycharmProjects/KaggleCompetitions/Playground_Sep2021/xgb_results2.csv")