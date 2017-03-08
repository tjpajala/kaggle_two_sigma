#Introduction
We participated in the [Kaggle Two Sigma data science competition](https://www.kaggle.com/c/two-sigma-financial-modeling) with the published scripts. The task was to predict returns of investment instruments with about 130 predictive variables. The competition was Kaggle's first Code Competition, meaning that submitted code was run on the organizer's server. Runtime was limited to 60 minutes, so we echewed the usage of intensive crossvalidation or feature selection techniques, and decided to rely on more computationally feasible methods.

The philosophy of both our scripts was similar: since the data has a low signal/noise ration, we decided on models that would not be very susceptible to overfitting. We supplemented the machine learning model with a simpler and more conservative regression model. Final predictions were always a combination of these two models. Below are more detailed descriptions of each of the script files.

##Extra trees with linear regression

This script downloads the training set and fits two models (i) extra trees with _n_ (pre-selected) features and linear regression with _m_ (pre-selected) features. Predictions are made with a linear combination of the two models, and prediction proceeds one batch at a time (approximately 1000 observations / batch). The script has no cross-validation or feature selection.

##Regression ensemble with differencing

This script downloads the training set, does 1st order differencing to each variable and then fits gradient boosting regression and ridge regression to the data. Differencing is done separately for each asset-ID (in both training and in test), and features are selected based on correlation analysis. Predictions are made with an ensemble of the two models: The ensemble is a linear regression, whose coefficients are estimated with a different dataset than the one the individual models were fitted with.