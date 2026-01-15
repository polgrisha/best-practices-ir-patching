from pingouin import partial_corr
import numpy as np
from scipy.stats import spearmanr, kendalltau
import pandas as pd
from tqdm import tqdm

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def output_partial_correlations(properties, target, data, features_to_control = ['score_delta']):
    correlations = dict()
    for property in properties:
        if property not in features_to_control:
            spearman_partial = partial_corr(data=data, x=target, y=property, covar=features_to_control, method='spearman')
            correlations[property] = {'spearman_partial': np.round(spearman_partial['r'].values[0], 3), 
                                      'p-value': np.round(spearman_partial['p-val'].values[0], 3)}
    return pd.DataFrame(correlations)


# def train_catboost_model(df, target_column, feature_columns):
#     X = df[feature_columns]
#     y = df[target_column]
    
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
    
#     model = CatBoostRegressor(
#         iterations=1000,
#         learning_rate=0.1,
#         depth=6,
#         loss_function='RMSE',
#         eval_metric='RMSE',
#         random_seed=42,
#         early_stopping_rounds=20,
#         verbose=100
#     )
    
#     model.fit(
#         X_train, y_train,
#         eval_set=(X_test, y_test),
#         plot=True
#     )
    
#     y_pred = model.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     r2 = r2_score(y_test, y_pred)
    
#     feature_importance = pd.DataFrame({
#         'feature': feature_columns,
#         'importance': model.feature_importances_
#     }).sort_values('importance', ascending=False)
    
#     return {
#         'model': model,
#         'rmse': rmse,
#         'r2': r2,
#         'feature_importance': feature_importance,
#         'predictions': y_pred
#     }
    
    
def train_catboost_model(df, target_column, feature_columns, n_runs=30, base_seed=42, test_size=0.2): 
    X = df[feature_columns]
    y = df[target_column]
    
    run_importances = []
    run_rmse = []
    run_r2 = []
    models = []

    for i in tqdm(range(n_runs)):
        seed = base_seed + i

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=seed,
            early_stopping_rounds=20,
            verbose=False
        )

        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            plot=False
        )

        y_pred = model.predict(X_test)
        run_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        run_r2.append(r2_score(y_test, y_pred))
        run_importances.append(model.feature_importances_)
        models.append(model)

    run_importances = np.array(run_importances)

    feature_importance_summary = pd.DataFrame({
        'feature': feature_columns,
        'importance_mean': run_importances.mean(axis=0),
        'importance_std': run_importances.std(axis=0, ddof=1)
    }).sort_values('importance_mean', ascending=False)

    return {
        'models': models,
        'rmse_mean': float(np.mean(run_rmse)),
        'rmse_std': float(np.std(run_rmse, ddof=1)),
        'r2_mean': float(np.mean(run_r2)),
        'r2_std': float(np.std(run_r2, ddof=1)),
        'feature_importance_summary': feature_importance_summary,
        'run_rmse': run_rmse,
        'run_r2': run_r2
    }