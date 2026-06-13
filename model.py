import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (r2_score,
                              mean_absolute_error,
                              mean_squared_error)
from xgboost import XGBRegressor
import pickle, os

FEATURES = [
    'age', 'km_driven', 'km_per_year',
    'age_km_interaction', 'is_low_mileage',
    'owner_num', 'engine_cc', 'max_power_bhp',
    'seats', 'fuel_type', 'transmission',
    'brand_tier', 'brand', 'seller_type'
]

CAT_COLS = [
    'fuel_type', 'transmission',
    'brand_tier', 'brand', 'seller_type'
]

def prepare_features(df):
    data   = df.copy()
    le_map = {}

    for col in CAT_COLS:
        if col in data.columns:
            le        = LabelEncoder()
            data[col] = le.fit_transform(
                            data[col].astype(str))
            le_map[col] = le

    available = [f for f in FEATURES
                 if f in data.columns]
    X = data[available]
    y = np.log1p(data['selling_price'])
    return X, y, le_map, available


def train_model(df):
    X, y, le_map, available_features = \
        prepare_features(df)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y,
                         test_size=0.2,
                         random_state=42)

    model = XGBRegressor(
        n_estimators    = 1000,
        learning_rate   = 0.05,
        max_depth       = 7,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        random_state    = 42,
        early_stopping_rounds = 50,
        eval_metric     = 'rmse',
        verbosity       = 0
    )

    model.fit(
        X_train, y_train,
        eval_set    = [(X_test, y_test)],
        verbose     = False
    )

    y_pred  = model.predict(X_test)
    y_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred)

    metrics = {
        'r2':   round(r2_score(y_test, y_pred), 4),
        'mae':  round(mean_absolute_error(
                    y_actual, y_pred_actual), 0),
        'rmse': round(np.sqrt(mean_squared_error(
                    y_actual, y_pred_actual)), 0),
    }

    # Save model and encoders
    with open('model_artifacts.pkl', 'wb') as f:
        pickle.dump({
            'model':    model,
            'le_map':   le_map,
            'features': available_features
        }, f)

    return model, metrics, le_map, available_features


def predict_price(input_dict):
    with open('model_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)

    model    = artifacts['model']
    le_map   = artifacts['le_map']
    features = artifacts['features']

    input_df = pd.DataFrame([input_dict])

    # Engineer features
    input_df['age'] = 2026 - input_df.get('year', 2020)
    input_df['km_per_year'] = (
        input_df['km_driven'] /
        input_df['age'].replace(0, 0.5))
    input_df['age_km_interaction'] = (
        input_df['age'] * input_df['km_driven'])
    input_df['is_low_mileage'] = (
        (input_df['km_per_year'] < 5000)
        .astype(int))

    for col, le in le_map.items():
        if col in input_df.columns:
            val = input_df[col].astype(str).iloc[0]
            if val in le.classes_:
                input_df[col] = le.transform([val])
            else:
                input_df[col] = le.transform(
                    [le.classes_[0]])

    X = input_df[features].fillna(0)
    log_pred = model.predict(X)[0]
    price    = np.expm1(log_pred)
    low      = price * 0.92
    high     = price * 1.08

    return {
        'predicted': round(price, 0),
        'low':       round(low, 0),
        'high':      round(high, 0)
    }