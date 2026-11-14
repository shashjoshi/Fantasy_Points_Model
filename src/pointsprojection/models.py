import pandas as pd
import numpy as np
import xgboost as xgb
from pointsprojection import evaluate, features


def buildxgboost(training_rows, val_rows, feat_cols, tar_col):
    #Initialize Regressor
    model = xgb.XGBRegressor(
        n_estimators = 1000,
        learning_rate = 0.01,
        max_depth = 5,
        objective = 'reg:squarederror',
        random_state = 42,
        early_stopping_rounds = 30,
    )
    model.fit(
        
        verbose=False
    )
        
    
def buildridgeregression(training_rows, val_rows, feat_cols, tar_col):
    pass

def buildneuralnet(training_rows, val_rows, feat_cols, tar_col): 
    pass

def buildbaseline(training_rows, val_rows, feat_cols, tar_col):
    pass
    