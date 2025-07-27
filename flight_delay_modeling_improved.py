# Improved Flight Delay Prediction Script with Enhanced Features and Tuning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor
import xgboost as xgb
from scipy.stats import randint, uniform

# Directories
export_dir = "./exportedVisualizations_v2"
model_dir = "./trained_models_v2"
prediction_dir = "./model_predictions_v2"
os.makedirs(export_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(prediction_dir, exist_ok=True)

# Load data
routes_df = pd.read_csv('./routes_data/routes_new_final.csv')
airports_df = pd.read_csv('./airports.csv')

# Merge airport data
routes_df = routes_df.merge(
    airports_df.add_prefix("origin_"), left_on="origin", right_on="origin_airport_code", how="left"
).merge(
    airports_df.add_prefix("dest_"), left_on="dest", right_on="dest_airport_code", how="left"
)

routes_df = routes_df.loc[:, ~routes_df.columns.str.startswith('Unnamed')]
routes_df.dropna(inplace=True)
routes_df['fl_date'] = pd.to_datetime(routes_df['fl_date'])

# Sort for calculating lag features
routes_df.sort_values(['op_unique_carrier', 'origin', 'dest', 'fl_date', 'crs_dep_time'], inplace=True)

# Add time-based features
def dep_bin(t):
    t = int(t)
    if t < 600: return 'early_morning'
    elif t < 1200: return 'morning'
    elif t < 1700: return 'afternoon'
    elif t < 2100: return 'evening'
    else: return 'night'

routes_df['dep_bin'] = routes_df['crs_dep_time'].apply(dep_bin)
routes_df['week_of_year'] = routes_df['fl_date'].dt.isocalendar().week
routes_df['is_summer'] = routes_df['week_of_year'].between(22, 36).astype(int)
routes_df['is_holiday'] = routes_df['fl_date'].dt.strftime('%m-%d').isin(['12-24','12-25','11-24','07-04']).astype(int)

# New lag feature: Previous delay for same OD-carrier
routes_df['prev_same_od_airline_delay'] = (
    routes_df.groupby(['op_unique_carrier', 'origin', 'dest'])['departure_delay']
    .shift(1)
)
routes_df['prev_same_od_airline_delay'].fillna(0, inplace=True)

# Define targets and features
y = np.log1p(routes_df['departure_delay'].clip(lower=0))  # log-transform positive delays

categorical = ['op_unique_carrier', 'dep_bin', 'day_of_week', 'origin_cluster_id', 'dest_cluster_id']
numerical = [
    'origin_avg_daily_flights', 'origin_percent_delayed', 'origin_num_unique_airlines',
    'origin_mean_dep_delay', 'origin_std_dep_delay',
    'dest_avg_daily_flights', 'dest_percent_delayed', 'dest_num_unique_airlines',
    'dest_mean_dep_delay', 'dest_std_dep_delay',
    'is_summer', 'is_holiday', 'prev_same_od_airline_delay'
]

X = routes_df[categorical + numerical]

# Transformers
preprocessor = ColumnTransformer([
    ('num', Pipeline([('scaler', StandardScaler()), ('pt', PowerTransformer())]), numerical),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])

# LightGBM tuning
lgb_params = {
    'model__n_estimators': randint(100, 300),
    'model__max_depth': randint(3, 10),
    'model__learning_rate': uniform(0.01, 0.1),
    'model__subsample': uniform(0.7, 0.3),
    'model__colsample_bytree': uniform(0.7, 0.3)
}

lgb_pipeline = Pipeline([
    ('pre', preprocessor),
    ('model', LGBMRegressor(random_state=42, n_jobs=-1))
])

search = RandomizedSearchCV(
    lgb_pipeline,
    param_distributions=lgb_params,
    n_iter=10,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='neg_root_mean_squared_error',
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Fit model
search.fit(X, y)
best_model = search.best_estimator_

# Save and evaluate
joblib.dump(best_model, os.path.join(model_dir, "LightGBM_Tuned.joblib"))
preds = best_model.predict(X)
residuals = y - preds

# Evaluation
metrics = {
    'RMSE': np.sqrt(mean_squared_error(y, preds)),
    'MAE': mean_absolute_error(y, preds),
    'R²': r2_score(y, preds)
}
print("Final LightGBM Metrics (Log-Transformed Delay):")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Save predictions to CSV
prediction_df = routes_df[['fl_date', 'origin', 'dest', 'op_unique_carrier']].copy()
prediction_df['true_delay'] = routes_df['departure_delay']
prediction_df['predicted_delay'] = np.expm1(preds)  # Inverse of log1p

prediction_output_path = os.path.join(prediction_dir, "lightgbm_tuned_predictions.csv")
prediction_df.to_csv(prediction_output_path, index=False)
print(f"\nPredictions saved to {prediction_output_path}")

# Residuals
plt.figure(figsize=(6, 4))
sns.histplot(residuals, bins=50, kde=True)
plt.title("Residual Distribution - LightGBM (Improved)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{export_dir}/residuals_LightGBM_Improved.png")
plt.close()

# Feature importance
if hasattr(best_model.named_steps['model'], 'feature_importances_'):
    importances = best_model.named_steps['model'].feature_importances_
    feature_names = best_model.named_steps['pre'].get_feature_names_out()
    sorted_idx = np.argsort(importances)[-15:]
    plt.figure(figsize=(6, 4))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.xlabel("Importance")
    plt.title("Top Feature Importances - LightGBM (Improved)")
    plt.tight_layout()
    plt.savefig(f"{export_dir}/feature_importance_LightGBM_Improved.png")
    plt.close()

