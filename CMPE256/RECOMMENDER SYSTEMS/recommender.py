import pandas as pd

from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import KFold,GridSearchCV
import os

file_path = os.path.expanduser('datasets/train.csv')

#training_df=pd.read_csv('datasets/train.csv')
test_df=pd.read_csv('datasets/test.csv',sep=",",skiprows=0)
# reader = Reader()
# data = Dataset.load_from_file(file_path, reader=reader)
# data = Dataset.load_from_df(training_df[['User', 'Track', 'Rating']], reader)

reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 100),skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)

from surprise import SVD

param_grid = {'n_epochs': [1,2,3,4,5,6,7,8,9,10], 'lr_all': [0.001,0.002,0.003,0.004,0.005],
              'reg_all': [0.1,0.2,0.3,0.4,0.6]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

algo = gs.best_estimator['rmse']
algo.fit(data.build_full_trainset())

preds={}

for index, row in test_df.iterrows():
    usr_id=str(row['User'])
    track_id=str(row['Track'])
    key=usr_id + "-"+ track_id
    preds[key]=int(algo.predict(usr_id,track_id).est)

result_df=pd.DataFrame(list(preds.items()),columns=['Id', 'Rating'])

# print(result_df)
result_df.to_csv('predictions.csv',index=False)