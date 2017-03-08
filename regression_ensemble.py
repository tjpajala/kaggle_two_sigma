#code for the API loop
import kagglegym
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
import matplotlib.pyplot as plt
import itertools
from sklearn.decomposition import PCA
from scipy.cluster.vq import kmeans2
import time
import math

#measure start time
start_time = time.time()
# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()
#observation.train.head(2)

#method for computing correlation, faster than corrcoef
def correlation_coefficient(x1, x2):
    product = np.mean((x1 - x1.mean()) * (x2 - x2.mean()))
    stds = x1.std() * x2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

unique_ids = pd.unique(observation.train.id)
#count nans per id
nans=dict(((x,np.sum(observation.train.loc[x,:].isnull(),axis=0)) for x in unique_ids))

#impute missing data in observation.train
md=observation.train.median(axis=0)
observation.train=observation.train.fillna(md,inplace=False)
print("Train imputation complete.")

#drop rows that are outliers
low_limit=-0.086093
high_limit=0.093497
dr=observation.train.shape[0]
print("Train is %g rows" % dr)
observation.train=observation.train[observation.train['y'] < high_limit]
observation.train=observation.train[observation.train['y'] > low_limit]
dq=dr-observation.train.shape[0]
print("Dropped %g rows" % dq)


col = [c for c in observation.train if c not in ["id","timestamp","y","label"]]

new_df = pd.DataFrame()
start_time = time.time()
#differencing for each id
for v in unique_ids:
    #get serie of an individual id
    serie = observation.train.loc[observation.train['id'] == v,col]
    
    serie = serie.tail(math.ceil(serie.shape[0]*0.30)) #only the last 30% for performance reasons
    
	#differencing with 1 period
    serie=serie.diff(periods=1,axis=0)
    serie['id']=v
    serie['y']=observation.train.loc[observation.train['id'] == v,'y'][1:]
	#drop NA's just to be sure (differencing creates them)
    serie = serie.dropna() 
    #append cut series together into a dataframe
    new_df = pd.concat([new_df, serie])

end_time = time.time()

print("Creating new_df took %g seconds" % (end_time - start_time))

col = [c for c in new_df if c not in ["id","timestamp","y","label"]]
corrs = pd.DataFrame(0, index=np.arange(len(unique_ids)), columns=col)
corrs = corrs.set_index(unique_ids)

start_time = time.time()
#computing correlations for each id
for v in unique_ids:
    #make series with id
    serie = new_df.loc[new_df['id'] == v]  
    
	#compute correlation between y and feature
    for c in col:
        #corrs.loc[v, c] = np.corrcoef(serie['y'],serie[c])[1,0]
        corrs.loc[v, c] = correlation_coefficient(serie['y'],serie[c])
    end_time = time.time()
    

corrs = corrs.fillna(0,inplace=False)
#save variable for number of NANs for each id
new_df['nans']=[nans[new_df.iloc[x]['id']] for x in range(len(new_df))]
print(new_df.head(2))
end_time = time.time()
print("computing all correlations/id took %g seconds" % (end_time - start_time))

#split into training and holdout for ensemble training
holdout=new_df.tail(math.ceil(new_df.shape[0]/2))
new_df=new_df.iloc[0:(math.ceil(new_df.shape[0]/2)-1)]

start_time = time.time()
#feature selection by highest average correlation with y
col=abs(corrs).mean(axis=0).sort_values(ascending=False,inplace=False).index.tolist()[:3]
col_w_nans=abs(corrs).mean(axis=0).sort_values(ascending=False,inplace=False).index.tolist()[:3]+['nans']
print(col)
#these are the three best variables
#col=['technical_30', 'technical_20', 'technical_19']

#Models
#GradientBoosting regressor
model1=GradientBoostingRegressor(loss="lad")
model1.fit(new_df[col_w_nans], new_df['y'])
#Ridge regression
model2=linear_model.RidgeCV(alphas=[0.001, 0.01, 0.1, 0.3],cv=10,scoring='r2')
model2.fit(new_df[col_w_nans], new_df['y'])
print(model2.alpha_)
print(model2.coef_)
print(model2.intercept_)
print(model1.score(new_df[col_w_nans], new_df['y']))
print(model2.score(new_df[col_w_nans], new_df['y']))


#stacking the two different models into ensemble
stacker= linear_model.LinearRegression()
stacker.fit(pd.DataFrame({"m1":model1.predict(holdout[col_w_nans]), "m2":model2.predict(holdout[col_w_nans])}), holdout['y'])
print(stacker.coef_)
print(stacker.intercept_)

#dictionary of the last observation of each ID
lastSeen=dict(((x, np.array(observation.train.loc[observation.train['id']==x,col].tail(1))) for x in unique_ids))
end_time = time.time()
print("FS and reg time was %g seconds" % (end_time - start_time))



while True:
    start_time = time.time()
    
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    test = observation.features
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        
    ids_test=pd.unique(test['id'])
    test.index=test['id']
	#save NANs of test frame
    test_nans=dict(((x,np.sum(test.loc[x,:].isnull(),axis=0)) for x in ids_test))
    #impute missing data by mean of train data
    test=test.fillna(md,inplace=False)
    
    #select only the best predictors
    test=test[col]
    end_time = time.time()
    #print("Test setup time was %g seconds" % (end_time - start_time))
    
    start_time = time.time()
	#set of the IDs we have already seen
    trainkeys=set(lastSeen.keys())
    
	#do numpy array for each test observation
    test_array=dict()
    for x in ids_test:
        rep=test.loc[x].values
        if x in trainkeys:
            #array is difference of new obs and last seen obs for the ID
            test_array[x]=np.append((rep-lastSeen[x]),test_nans[x]).reshape(1,-1)
        lastSeen[x]=rep
        
    end_time = time.time()
    #print("Test array creation time was %g seconds" % (end_time - start_time))
    
    
    start_time = time.time()
	#cutoff for predictions, this was estimated from previous analyses
    low_y_cut = -0.075
    high_y_cut = 0.075
    y_vals2=[np.asscalar(model2.predict(test_array[x])) if x in trainkeys else 0 for x in ids_test]
    y_vals1=[np.asscalar(model1.predict(test_array[x])) if x in trainkeys else 0 for x in ids_test]
    y_vals_stack=stacker.predict(pd.DataFrame({"m1":np.array(y_vals1), "m2":np.array(y_vals2)}))
    #target.y=np.clip(np.array(y_vals1),low_y_cut,high_y_cut)*0.35+np.clip(np.array(y_vals2),low_y_cut,high_y_cut)*0.65
    target.y=np.clip(np.array(y_vals_stack),low_y_cut,high_y_cut)
    #print(target.head(5))
    end_time = time.time()
    #print("Regression predict time was %g seconds" % (end_time - start_time))
    
	#step predictions, get new observation
    observation, reward, done, info = env.step(target)
        
    if done:
        print("Finished!", info["public_score"])
        break
    if timestamp % 100 == 0:
        print("Reward #{}".format(reward))