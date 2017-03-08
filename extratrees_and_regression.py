import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#set kagglegym environment
env = kagglegym.make()
o = env.reset()
excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
col1 = [c for c in o.train.columns if c not in excl]

#weight for model 1
m1_weight=0.4



train = o.train[col1]
#count number of NANs for each row
n = train.isnull().sum(axis=1)
#save median of each column for imputation
d_mean= train.median(axis=0)

#impute missing values with median
train = train.fillna(d_mean)
#save number of missing values
train['znull'] = n
#make new feature based on our analysis
train['drv']=train['technical_30']-train['technical_20']
#select only top 5 predictors
col=['technical_30','technical_20','technical_22','znull','drv']
train=train[col]

#turn 5 best features into 3rd degree polynomial features
pf=PolynomialFeatures(3)
train=pf.fit_transform(train)
n = []

#model1: ExtraTrees
rfr = ExtraTreesRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=308537, verbose=0)
model1 = rfr.fit(train, o.train['y'])

#remove outliers for regression model
#https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189
low_y_cut = -0.075
high_y_cut = 0.075
y_is_above_cut = (o.train.y > high_y_cut)
y_is_below_cut = (o.train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
#model2: basic linear univariate regressor
model2 = LinearRegression(n_jobs=-1)
model2.fit(np.array(o.train[col1].fillna(d_mean).loc[y_is_within_cut, 'technical_20'].values).reshape(-1,1), o.train.loc[y_is_within_cut, 'y'])

#function for regulating predictions by median of each ID
#https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659
ymean_dict = dict(o.train.groupby(["id"])["y"].median())
def get_weighted_y(series):
    id, y = series["id"], series["y"]
    return 0.95 * y + 0.05 * ymean_dict[id] if id in ymean_dict else y

while True:
    excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
    col1 = [c for c in o.features.columns if c not in excl]
    test = o.features[col1]
	#save number of NANs
    n = test.isnull().sum(axis=1)
    if o.features.timestamp[0] % 100 == 0:
        print("Timestamp #{}".format(o.features.timestamp[0]))
	#impute missing values with train median
    test = test.fillna(d_mean)
	#save number of missing values
    test['znull'] = n
	#do feature engineering
    test['drv']=test['technical_30']-test['technical_20']
	#select only five best observations
    col=['technical_30','technical_20','technical_22','znull','drv']
    test=test[col]
	#do 3rd degree polynomial transform for best predictors
    test=pf.fit_transform(test)
    pred = o.target
	#model2: univariate linear regressor
    test2 = np.array(o.features[col1].fillna(d_mean)['technical_20'].values).reshape(-1,1)
	#prediction: w1*model1 + (1-w1)*model2
    pred['y'] = (model1.predict(test).clip(low_y_cut, high_y_cut) * m1_weight) + (model2.predict(test2).clip(low_y_cut, high_y_cut) * (1-m1_weight))
	#regulation by median ID-specific returns
    pred['y'] = pred.apply(get_weighted_y, axis = 1)
    o, reward, done, info = env.step(pred)
    if done:
        print("Finished!", info["public_score"])
        break
    if o.features.timestamp[0] % 100 == 0:
        print("Reward #{}".format(reward))