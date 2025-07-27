import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor
import xgboost as xgb

export_dir = "./exportedVisualizations"
model_dir = "./trained_models"
prediction_dir = "./model_predictions"
os.makedirs(export_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(prediction_dir, exist_ok=True)

routes_df = pd.read_csv('./routes_data/routes_new_final.csv')
airports_df = pd.read_csv('./airports.csv')

routes_df = routes_df.merge(
    airports_df.add_prefix("origin_"), left_on="origin", right_on="origin_airport_code", how="left"
).merge(
    airports_df.add_prefix("dest_"), left_on="dest", right_on="dest_airport_code", how="left"
)

routes_df = routes_df.loc[:, ~routes_df.columns.str.startswith('Unnamed')]

pre_dropna_rowcount = len(routes_df)
routes_df.dropna(inplace=True)
post_dropna_rowcount = len(routes_df)

routes_df.head(10).to_csv("joined_routes_sample.csv", index=False)
print(f"Rows before dropna: {pre_dropna_rowcount}, after: {post_dropna_rowcount}, dropped: {pre_dropna_rowcount - post_dropna_rowcount}")

print("Sorting joined dataframe by flight date...")
sort_start_time = time.time()
routes_df.sort_values("fl_date", inplace=True)
print(f"Sorting finished. Sorting took {time.time() - sort_start_time} seconds.")

def dep_bin(t):
    t = int(t)
    if t < 600: return 'early_morning'
    elif t < 1200: return 'morning'
    elif t < 1700: return 'afternoon'
    elif t < 2100: return 'evening'
    else: return 'night'

routes_df['dep_bin'] = routes_df['crs_dep_time'].apply(dep_bin)
routes_df['fl_date'] = pd.to_datetime(routes_df['fl_date'])
routes_df['week_of_year'] = routes_df['fl_date'].dt.isocalendar().week

y = routes_df['departure_delay']
categorical = ['op_unique_carrier', 'dep_bin', 'day_of_week', 'origin_cluster_id', 'dest_cluster_id']
numerical = [
    'origin_avg_daily_flights', 'origin_percent_delayed', 'origin_num_unique_airlines',
    'origin_mean_dep_delay', 'origin_std_dep_delay',
    'dest_avg_daily_flights', 'dest_percent_delayed', 'dest_num_unique_airlines',
    'dest_mean_dep_delay', 'dest_std_dep_delay'
]
X = routes_df[categorical + numerical]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])

models = {
    'Linear Regression': LinearRegression(),
    # 'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Quantile Regression (alpha=0.9)': GradientBoostingRegressor(loss="quantile", alpha=0.9, n_estimators=100)
}

routes_df['split_week'] = routes_df['week_of_year']
splits = TimeSeriesSplit(n_splits=5)
results = {}
lightgbm_pipeline = None  # Keep for later

for name, model in models.items():
    print(f"\nTraining {name}")
    pipeline = Pipeline([
        ('pre', preprocessor),
        ('model', model)
    ])

    rmses, maes, r2s = [], [], []
    all_y, all_preds = [], []

    start_time = time.time()

    for fold_num, (train_idx, test_idx) in enumerate(splits.split(X)):
        print(f"  Fold {fold_num + 1}/5...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_start = time.time()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(f"    Fold trained in {time.time() - fold_start:.2f} seconds")

        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        maes.append(mean_absolute_error(y_test, y_pred))
        r2s.append(r2_score(y_test, y_pred))

        all_y.extend(y_test)
        all_preds.extend(y_pred)

    total_time = time.time() - start_time
    results[name] = {
        'RMSE': np.mean(rmses),
        'MAE': np.mean(maes),
        'R²': np.mean(r2s),
        'Training Time (s)': total_time
    }

    print(f"\nResults for {name}:")
    for k, v in results[name].items():
        print(f"  {k}: {v:.4f}")

    # Save model
    joblib.dump(pipeline, os.path.join(model_dir, f"{name.replace(' ', '_')}.joblib"))

    # Save predictions
    pd.DataFrame({'true': all_y, 'predicted': all_preds}).to_csv(
        os.path.join(prediction_dir, f"{name.replace(' ', '_')}_predictions.csv"), index=False)

    # Residual plot
    residuals = np.array(all_y) - np.array(all_preds)
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=50, kde=True)
    plt.title(f'Residual Distribution - {name}')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{export_dir}/residuals_{name.replace(" ", "_")}.png')
    plt.close()

    # Feature importances
    if hasattr(model, 'feature_importances_'):
        trained_model = pipeline.named_steps['model']
        importances = trained_model.feature_importances_
        feature_names = pipeline.named_steps['pre'].get_feature_names_out()
        sorted_idx = np.argsort(importances)[-15:]

        plt.figure(figsize=(6, 4))
        plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
        plt.xlabel('Importance')
        plt.title(f'Feature Importance - {name}')
        plt.tight_layout()
        plt.savefig(f'{export_dir}/feature_importance_{name.replace(" ", "_")}.png')
        plt.close()

        # XGBoost built-in
        if name == 'XGBoost':
            print("Plotting built-in XGBoost importance...")
            xgb_model = pipeline.named_steps['model']
            xgb.plot_importance(xgb_model, max_num_features=15)
            plt.tight_layout()
            plt.savefig(f"{export_dir}/xgboost_builtin_importance.png")
            plt.close()

    if name == 'LightGBM':
        lightgbm_pipeline = pipeline

# Final results
results_df = pd.DataFrame(results).T
print("\nAverage Evaluation Metrics:")
print(results_df)

results_df.to_csv("model_evaluation_results.csv")
results_df[['RMSE', 'MAE', 'R²']].plot(kind='bar', figsize=(10, 5), title='Model Evaluation Metrics')
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{export_dir}/model_comparison_metrics.png')
plt.close()

# Permutation Importance for LightGBM (only once)
if lightgbm_pipeline:
    print("\nCalculating permutation importance for LightGBM...")
    perm_result = permutation_importance(lightgbm_pipeline, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    perm_importances = perm_result.importances_mean
    feature_names = lightgbm_pipeline.named_steps['pre'].get_feature_names_out()
    sorted_idx = np.argsort(perm_importances)[-15:]

    plt.figure(figsize=(6, 4))
    plt.barh(np.array(feature_names)[sorted_idx], perm_importances[sorted_idx])
    plt.xlabel('Mean Importance')
    plt.title('Permutation Feature Importance - LightGBM')
    plt.tight_layout()
    plt.savefig(f"{export_dir}/permutation_importance_lightgbm.png")
    plt.close()

