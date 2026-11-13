#imports
import pandas as pd
import numpy as np
from pointsprojection import features
import sklearn.metrics as sklm

#Get training splits and return a list of (training row, validation row) pairs
#8 folds, with validation row Y having training rows 2016 to Y-1
#2025/26 held for out-of-sample testing later
def seasonsplits (feat_df, val_seasons = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]):
    retlist = [[],[]]
    for i in val_seasons:
        training = feat_df[feat_df['season'] < i].reset_index(drop = True)
        validation = feat_df[feat_df['season'] == i].reset_index(drop=True)
        retlist[0].append(training)
        retlist[1].append(validation)
    return retlist


def computemetrics (df, preds):
    #define target
    target = features.featurereg['target'][0]
    #Assign predictions to the df ans strip pandas indexing
    df = df.assign(pred = np.asarray(preds))
    #Calculate Mean Absolute Error 
    mae = sklm.mean_absolute_error(df['pred'], df[target])
    #Calculate Root Mean Squared Error
    rmse = sklm.root_mean_squared_error(df['pred'], df[target])
    #Calculate Directional Bias
    bias = (df['pred']-df[target]).mean()
    #Calculate MAE per position group
    maeperpos = df.groupby('position').apply(
        lambda x : sklm.mean_absolute_error(x['pred'], x[target])
    )
    #Calculate Spearman correlation between actual and predicted values
    corr = df.groupby(['season', 'week', 'position']).apply(
        lambda x: x[target].corr(x['pred'], method = 'spearman')
    ).mean()
    #Return Dictionary
    retdict = {
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'maeperpos': maeperpos,
        'corr': corr
    }
    return retdict

#Marks the realistic starters for each season, week, position
#Ex. top 36 QBs, Top 50 RBs, Top 50 WRs
realistic_cutoffs = {'QB':26, 'RB':50, 'WR':50, 'TE':36}
#Ranked by pre-game ppr_ewm
def startablemask(df):
    rank = df.groupby(['season', 'week', 'position'])['ppr_ewm'].rank(method='first',ascending=False)
    df['startable'] = rank <= df['position'].map(realistic_cutoffs)
    return df

#Scores a model to evaluate effectiveness
def crossvalidate(model_fn, features_df, feature_cols, target, val_seasons = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]):
    #Get folds plit by training and validation
    trainlist, vallist = seasonsplits(features_df, val_seasons)

    #Metrics dictionary and DataFrame per fold
    resultrows = []
    predframes = []
    for season, training_rows, val_rows in zip(val_seasons, trainlist, vallist):
        #train on fold and predict validation season
        preds = np.asarray(model_fn(training_rows, val_rows, feature_cols, target))
        #If wrong number of predictions, stop and raise
        if len(preds) != len(val_rows):
            raise ValueError(f"{season}: model returned {len(preds)} predictions for {len(val_rows)} rows")

        #Mark realistic starters with mask function
        val_rows = startablemask(val_rows.copy())
        startable = val_rows['startable'].to_numpy()

        #metrics on full pool and startable subset
        full = computemetrics(val_rows, preds)
        start = computemetrics(val_rows[startable], preds[startable])

        #One results row per fold, MAE by position split into their own columns
        row = {'season': season, 'n_rows': len(val_rows), 'n_startable': int(startable.sum())}
        #Copies metrics into a row 
        for name, m in (('all', full), ('start', start)):
            row[f'{name}_mae'] = m['mae']
            row[f'{name}_rmse'] = m['rmse']
            row[f'{name}_bias'] = m['bias']
            row[f'{name}_spearman'] = m['corr']
            for pos, val in m['maeperpos'].items():
                row[f'{name}_mae_{pos}'] = val
        #Save the finished results for that fold
        resultrows.append(row)

        #Keep every validation row with it's prediction for error analysis
        predframes.append(val_rows.assign(pred = preds))

    #Turn list of dicts into a table
    results = pd.DataFrame(resultrows)
    #Average across folds as a final row
    avg = results.drop(columns = 'season').mean(numeric_only = True)
    avg['season'] = 'mean'
    results = pd.concat([results, avg.to_frame().T], ignore_index = True).infer_objects()

    oof_preds = pd.concat(predframes, ignore_index = True)
    return results, oof_preds
        

def comparemodels(results):
    pass

def finaltest(model_fn, features_df, feature_cols, target, test_seasons):
    pass

