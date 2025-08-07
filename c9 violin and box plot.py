# July 5, 2025
# Pairplot of different features with malignancy

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set style for better visualization
plt.style.use('default')
sns.set_palette("husl")

code_outputs = 'code outputs/c9 violin and box plot'
os.makedirs(code_outputs, exist_ok=True)

# ----------------------------
# Load and preprocess data
# ----------------------------
df = pd.read_excel("LIDC-IDRI nodule metadata.xlsx")
df = df[df['malignancy'] != 3] # remove uncertain nodules
df = df[df['nodule selected'] != 'No'] # keep only the eligible nodules
df['label'] = df['malignancy'].apply(lambda x: 1 if x >= 4 else 0)

df[['centroid_x', 'centroid_y', 'centroid_z']] = df['nodule centroid'].apply(
    lambda x: pd.Series(eval(x) if isinstance(x, str) else x)
)

print("Available columns:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)


# Separate numerical and categorical features
numerical_features = [
    'nodule diameter (mm)', 'centroid_x', 'centroid_y', 'centroid_z'
]

categorical_features = [
    'subtlety', 'internal structure', 'calcification', 'sphericity',
    'margin', 'lobulation', 'spiculation', 'textures'
]

# Check data types and handle mixed features
print("\n=== Feature Analysis ===")
print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)

# Create feature dataframe with proper handling
feature_df = df[numerical_features + categorical_features].copy()

# Convert numerical features to float
for feature in numerical_features:
    if feature in df.columns:
        feature_df[feature] = pd.to_numeric(df[feature], errors='coerce')

# Handle categorical features - convert to numeric if possible, otherwise keep as categorical
for feature in categorical_features:
    if feature in df.columns:
        # Try to convert to numeric, if fails keep as categorical
        try:
            feature_df[feature] = pd.to_numeric(df[feature], errors='coerce')
            if feature_df[feature].isna().sum() > 0:
                print(f"Warning: {feature} has missing values after conversion")
        except:
            print(f"Keeping {feature} as categorical")
            feature_df[feature] = df[feature].astype('category')

print("\nFeature dataframe info:")
print(feature_df.info())


# ----------------------------
# Create binary malignancy classification
# ----------------------------
# Create a dataframe for analysis with binary malignancy
plot_df = feature_df.copy()
plot_df['malignancy_binary'] = df['malignancy'].apply(lambda x: 1 if x >= 4 else 0)  # 1=malignant, 0=benign
plot_df['malignancy_label'] = plot_df['malignancy_binary'].apply(lambda x: 'Malignant' if x == 1 else 'Benign')

# Get numerical features for correlation analysis
numerical_features_available = [f for f in numerical_features if f in plot_df.columns]
categorical_features_available = [f for f in categorical_features if f in plot_df.columns]

print(f"\nNumerical features available: {numerical_features_available}")
print(f"Categorical features available: {categorical_features_available}")

key_numerical_features = numerical_features
key_categorical_features = categorical_features

key_features = key_numerical_features + key_categorical_features
print(f"Key features for analysis: {key_features}")



# ----------------------------
# Create individual feature vs malignancy plots
# ----------------------------
# Calculate number of rows and columns for subplot layout
n_features = len(key_features)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 7))
if n_features == 1:
    axes = [axes]
elif n_rows == 1:
    axes = axes.reshape(1, -1)
else:
    axes = axes.ravel()

for i, feature in enumerate(key_features):
    if i < len(axes):
        ax = axes[i]
        
        # Check if feature is numerical or categorical
        if plot_df[feature].dtype in ['float64', 'int64']:
            # Numerical feature - use box plot and violin plot
            try:
                # Box plot
                bp = ax.boxplot([plot_df[plot_df['malignancy_binary'] == 0][feature], 
                               plot_df[plot_df['malignancy_binary'] == 1][feature]], 
                              tick_labels=['Benign', 'Malignant'], patch_artist=True)
                
                # Color the boxes
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][1].set_facecolor('lightcoral')
                
                # Add violin plot overlay
                violin_parts = ax.violinplot([plot_df[plot_df['malignancy_binary'] == 0][feature], 
                                           plot_df[plot_df['malignancy_binary'] == 1][feature]], 
                                          positions=[1, 2])
                
                # Color the violin plots and set alpha
                violin_parts['bodies'][0].set_facecolor('lightblue')
                violin_parts['bodies'][0].set_alpha(0.3)
                violin_parts['bodies'][1].set_facecolor('lightcoral')
                violin_parts['bodies'][1].set_alpha(0.3)
                
                # Set labels and title
                # ax.set_xlabel('Malignancy', fontsize=12)
                ax.set_ylabel(feature, fontsize=12)
                # ax.set_title(f'{feature} vs Binary Malignancy', fontsize=14, fontweight='bold')
                
                # Add correlation coefficient
                corr = plot_df[feature].corr(plot_df['malignancy_binary'])
                ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                        transform=ax.transAxes, fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except Exception as e:
                print(f"Error plotting numerical feature {feature}: {e}")
                ax.text(0.5, 0.5, f'Error plotting {feature}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{feature} - Error', fontsize=14, fontweight='bold')
                
        else:
            # Categorical feature - use count plot
            try:
                # Create count plot
                sns.countplot(data=plot_df, x=feature, hue='malignancy_label', ax=ax)
                ax.set_title(f'{feature} vs Binary Malignancy', fontsize=14, fontweight='bold')
                ax.set_xlabel(feature, fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                
                # Rotate x-axis labels if they're too long
                if len(feature) > 15:
                    ax.tick_params(axis='x', rotation=45)
                
                # Add chi-square test or other categorical association measure
                from scipy.stats import chi2_contingency
                contingency_table = pd.crosstab(plot_df[feature], plot_df['malignancy_binary'])
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                ax.text(0.05, 0.95, f'Chi² p-value: {p_value:.3f}', 
                       transform=ax.transAxes, fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except Exception as e:
                print(f"Error plotting categorical feature {feature}: {e}")
                ax.text(0.5, 0.5, f'Error plotting {feature}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{feature} - Error', fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)

# Hide empty subplots
for i in range(len(key_features), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig(f'{code_outputs}/individual_feature_malignancy_plots.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ----------------------------
# Print summary statistics
# ----------------------------
print("\n=== Feature vs Binary Malignancy Analysis ===")
print(f"Total samples: {len(plot_df)}")
print(f"Benign samples (0): {len(plot_df[plot_df['malignancy_binary'] == 0])}")
print(f"Malignant samples (1): {len(plot_df[plot_df['malignancy_binary'] == 1])}")

print("\n=== Numerical Feature Correlation with Binary Malignancy ===")
for feature in key_features:
    if plot_df[feature].dtype in ['float64', 'int64']:
        corr = plot_df[feature].corr(plot_df['malignancy_binary'])
        print(f"{feature}: {corr:.3f}")

print("\n=== Categorical Feature Association with Binary Malignancy ===")
for feature in key_features:
    if plot_df[feature].dtype not in ['float64', 'int64']:
        try:
            from scipy.stats import chi2_contingency
            contingency_table = pd.crosstab(plot_df[feature], plot_df['malignancy_binary'])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            print(f"{feature}: Chi² p-value = {p_value:.3f}")
        except:
            print(f"{feature}: Could not calculate association")

print(f"\nPlots saved in: {code_outputs}")

