import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
pred_path = './model_predictions_v2/lightgbm_tuned_predictions.csv'
export_dir = './exportedVisualizations_v2'
os.makedirs(export_dir, exist_ok=True)

# Load predictions
df = pd.read_csv(pred_path)

# Calculate absolute error
df['abs_error'] = (df['true_delay'] - df['predicted_delay']).abs()

# Compute detailed statistics
stats = {
    'Mean Absolute Error (MAE)': df['abs_error'].mean(),
    'Median Absolute Error': df['abs_error'].median(),
    '90th Percentile Absolute Error': df['abs_error'].quantile(0.90),
    '95th Percentile Absolute Error': df['abs_error'].quantile(0.95),
    '99th Percentile Absolute Error': df['abs_error'].quantile(0.99),
    'Maximum Absolute Error': df['abs_error'].max(),
    'Minimum Absolute Error': df['abs_error'].min(),
    'Standard Deviation of Absolute Error': df['abs_error'].std(),
    'Total Predictions': len(df),
    'Proportion with < 15 min error': (df['abs_error'] < 15).mean(),
    'Proportion with < 30 min error': (df['abs_error'] < 30).mean(),
    'Proportion with < 60 min error': (df['abs_error'] < 60).mean(),
}

# Print to console
print("\nAbsolute Error Statistics:")
for k, v in stats.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# Save statistics to file
with open(os.path.join(export_dir, "abs_error_statistics.txt"), 'w') as f:
    for k, v in stats.items():
        f.write(f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")

# Plot absolute error distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['abs_error'], bins=50, kde=True)
plt.title("Absolute Error Distribution - LightGBM")
plt.xlabel("Absolute Error (minutes)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(export_dir, "abs_error_distribution_LightGBM.png"))
plt.show()

