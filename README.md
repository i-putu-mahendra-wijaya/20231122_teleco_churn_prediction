# Predicting Customer Churn

The data for the model is obtained from [Kaggle](https://urldefense.com/v3/__https:/www.kaggle.com/datasets/blastchar/telco-customer-churn__;!!EIXh2HjOrYMV!fOQ3CPiCQm8Fqpck5y0KPqeJfnirgV7ZQ4QCxYdEqDPaEPQJZJ9JYmfJ0YAim1QdrxmFDlXh0__IrLgKJigGc1TJSB3Gjg$)

## Data Preparation & Cleaning

Exploring, treating, and cleaning raw data can be found in the `notebooks` folder

```
notebooks/exploration/part00_data_exploration.ipynb 
```

In this notebook, we try to explore the raw data: checking for outliers, checking null values, each variable distribution, and 
converting variables to appropriate types 

Several key findings from this notebook are: 
* There are no outlier detected
* The column `TotalCharges` is marked as categorical while it was supposed to be numerical
* Converting `TotalCharges` column to numerical introduces 11 `null` values
* The column `SeniorCitizen` is marked as numerical while it was supposed to be categorical
* The `Churn` target column is unbalanced

Some columns too are observed to have unbalanced distribution. 

These unbalanced distribution will introduce bias which will need to be checked against later

As with regard to `Churn` target column, we also do augmentation to ameliorate the distribution from the original 
5-to-2 to roughly 1-to-1 No : Yes 

## Dataset Splitting

Splitting dataset, and calculating mean from `training` dataset to impute `null` values can 
be found in `notebook` folder

```
notebooks/exploration/part01_data_splitting.ipynb
```

After splitting the dataset to `training`, `validation`, and `testing`; we calculate the mean `total_charges` 
taken from the `testing` dataset. 

We then impute the `null` in `TotalCharge` column from this calculated mean. 

Why use calculated mean only from `training` dataset? Why not use calculated mean from the entire dataset?

Because we want to avoid "leakage" of information to `validation` and `testing` dataset. 

Hence, by the same logic, we will impute all `null` found in `validation` and `testing` dataset with the 
mean calculated only from `testing` dataset. 

In this section, we also convert all categorical columns in the dataset to `one-hot-encoded`` format. 

This step results in explosion on the number of column from the initial 19 columns 
(after excluding `customerID` and `Churn`) to 46 columns

Furthermore, since one-hot-encoding expand the column by putting `1` and `0`, it also make the resulting 
matrix very sparse, which is very inefficient and hard to calculate


## Feature Engineering

Converting one-hot encoded features into its PCA can be found in `notebook` folder 

```
notebooks/exploration/part02_feature_engineering.ipynb
```

In this section, we will further pre-process the data before we jump into the model

As noted earlier, the application of `one-hot-encoding` to convert categorical variables to numerical variables
are required before the raw data can be used for classification model

However, this step also produce sparse matrix: that is, matrix that are predominated by `1` and `0`

Hence, in this section, we will reduce the dimensionality of the data from `46` to `3` using 
Primary Component Analysis (PCA). 

Briefly PCA is a mathematical technique which aims to find the best projection from higher-dimension to lower-dimension:
in our case from 46 original dimension to smaller dimension.

Naturally, the question to ask is what is the destination lower dimension then? 

As a rule of thumb, we want to reduce the dimension to fourth root of the original: which means for us (46)^(0.25) ~ 3


## Algorithm / implementation section, Model Training, and Performance Measurement

Implementing four classifier algorithm candidates: 
1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Support Vector Classifier

The implementation and training result can be found in `notebook` folder 

``` 
notebooks/exploration/part03_algo_implementation.ipynb
```

In this section, we are exploring potential model candidates for the classification task at hand using
default hyperparameters from scikit-learn

The aim here is just to find the best candidate, not yet optimizing the parameter

There are several intersting points to note from this simple experiment

1. `Logistic Regression` model, being the simplest models of all, have the shortest training time, and it already offers reasonable accuracy at ~ 0.7 throughout traiining, validation, and testing. Furthermore, comparing `accuracy_score` and `f1_score` across training, validation, and testing stages gives a hint that `Logistic Regression` - despite its simplicity - ca be generalized. That is, the model does not over fit. That said, `Logistic Regression` model have only few hyparparameter that can be tuned - and hence, it may be difficult to fine-tune the model further
2. `Random Forest Classifier` model have the best overall accuracy amongst all other models. The training time of `Random Forest Classifier` is about 10x slower than Logistic Regression, meaning it considerably more complex than Logistic Regression. This training time, however, is still relatively modest if compared to  `Support Vector Classifier` model. Comparing the accuracy and f1_score metrics in training, validation, and testing stage, however, reveals that `Random Forest Classifier` grossly over fit, and hence - as it is - will not be as usable if we feed new data to predict classification. `Random Forest Classifier` have many more hyperparameters that can be tuned compared to `Logistic Regression`, which means we may be able to reduce the over fitting problem by appropriately fine-tuning the model
3. `Gradient Boosting Classifier` take slightly longer to train compared to `Random Forest Classifier`. This is expected, given that the algorightm is using sequences of weak learners to rectify errors made from earlier weak learners. This longer training time, however, seems to be worth it, because comparing the metrics accuracy and f1_score across training, validation, and testing stages seems to indicate that the model is actually generalizable - and hence, should perform better compared to `Random Forest Classifier` when fed new data for classification task. Furthermore, we have not yet fine tune the model in this step, and as such the performance given is given out-of-the-box from the default settings from sci-kit learn. Hence, `Gradient Boosting Classifier` seems to give us a bright promise to fine tune further.
4. Finally, `Support Vactor Classifier` takes the longest time tom train. This is too expected given that the strategy taken with this algorithm is to project the data points to every higher dimension to search a hyperplane that can separate the data into appropriate class. This complexity, however, does not give us extra benefit, as can be seen from the accuracy and f1_score across training, validation, and testing stage. 


Hence, we will select `Gradient Boosting Classification` algorithm to base our hyperparameter tuning later.

In this section, we also explore the model explainability using `SHAP` value calculation.

The result from `SHAP` value calculation will be useful for us to zoom-in on which bias we need to focus on 
in later stage


## Hyperparameter Tuning

From all the model candidate, `GradientBoostingClassifier` model was chosen because it balanced between complexity
and accuracy. In the model selection, the parameters used were the defaults out-of-the box from scikit-learn

In `hyperparametr tuning` section, we start to tune the hyperparameters for `GradientBoostingClassifier` 

The list of hyperparameters that were tested are as follows: 

* n_estimators: [100, 200, 300] >>> number of boosting tree
* learning_rate: [0.1, 0.2, 0.3] >>> learning rate
* max_depth: [3, 5, 7] >>> maximum depth in each tree
* min_samples_split: [2, 5, 10] >>> minimum number of samples required to split a node

The notebook for finding the best parameters can be found in `notebook` folder

``` 
./notebooks/exploration/part04_hyperparameter_tuning.ipynb
```

After searching for the best hyperparameter using scikit-learn `GridSearchCV`, the following hyper-param 
are found to results in best `GradientBoostingClassifier`

```
{   
    'learning_rate': 0.2,
    'max_depth': 7,
    'min_samples_split': 2,
    'n_estimators': 200
}
```

Further, the accuracy of the best_model found when subjeced to train, validate, and testing dataset

```
Best GradientBoostingClassifier model
search period 776.6343929767609 seconds
----------------------------------------
Accuracy Score train 0.9942012719790497
Accuracy Score val 0.8126752664049355
Accuracy Score test 0.7986539540100953

```
The model is over-fitting quite a bit

## Model Fairness / Bias

The result from `SHAP` value calculation can be seen from `notebook/figures` folder

``` 
notebooks/figures/shap_value_explainer.png
```

![shap value](notebooks/figures/shap_value_explainer.png)

The picture show Top 20 features that most influence the output of `GradientBoostingClassifier` and its
direction of influence. 

The picture, of course, is still difficult to interpret because it only gives us index of the feature

Here's the map that convert the index back to meaningful names

``` 
feature_0 is tenure
feature_1 is MonthlyCharges
feature_2 is TotalCharges
feature_3 is gender_Female
feature_4 is gender_Male
feature_5 is SeniorCitizen_0
feature_6 is SeniorCitizen_1
feature_7 is Partner_No
feature_8 is Partner_Yes
feature_9 is Dependents_No
feature_10 is Dependents_Yes
feature_11 is PhoneService_No
feature_12 is PhoneService_Yes
feature_13 is MultipleLines_No
feature_14 is MultipleLines_No phone service
feature_15 is MultipleLines_Yes
feature_16 is InternetService_DSL
feature_17 is InternetService_Fiber optic
feature_18 is InternetService_No
feature_19 is OnlineSecurity_No
feature_20 is OnlineSecurity_No internet service
feature_21 is OnlineSecurity_Yes
feature_22 is OnlineBackup_No
feature_23 is OnlineBackup_No internet service
feature_24 is OnlineBackup_Yes
feature_25 is DeviceProtection_No
feature_26 is DeviceProtection_No internet service
feature_27 is DeviceProtection_Yes
feature_28 is TechSupport_No
feature_29 is TechSupport_No internet service
feature_30 is TechSupport_Yes
feature_31 is StreamingTV_No
feature_32 is StreamingTV_No internet service
feature_33 is StreamingTV_Yes
feature_34 is StreamingMovies_No
feature_35 is StreamingMovies_No internet service
feature_36 is StreamingMovies_Yes
feature_37 is Contract_Month-to-month
feature_38 is Contract_One year
feature_39 is Contract_Two year
feature_40 is PaperlessBilling_No
feature_41 is PaperlessBilling_Yes
feature_42 is PaymentMethod_Bank transfer (automatic)
feature_43 is PaymentMethod_Credit card (automatic)
feature_44 is PaymentMethod_Electronic check
feature_45 is PaymentMethod_Mailed check
```
Some interesting points to note from this `SHAP` value: 

1. The top `5` most important features to predict `churn` are all categorical: 
   * TechSupport_No
   * OnlineSecurity_No
   * Contract_Month-to-Month
   * InternetService_Fiber optic
   * OnlineBackup_No

2. There are only two numerical variables appear in this Top 20: 
   * `MonthlyCharges`, appearing in No.10. The SHAP value indicates that customer with high `MonthlyCharges` have more tendency to churn
   * `tenure`, appearing in No.20. The SHAP value indicates that customer with young `tenure`  is less likely to churn

## Pushing Models to Production

### Key Stages & Processes

### What Tools to Leverage

### What to Monitor Whilst In Production

## Business / Commercial Need for Alignment with Analytics Team Prior to Retention Marketing Campaign