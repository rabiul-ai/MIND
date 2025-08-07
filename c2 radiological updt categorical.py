import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, log_loss
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import clone
import xgboost as xgb
import matplotlib.cm as cm

code_outputs = 'code outputs/c2 radiological updt categorical 955'

# ----------------------------
# Load and preprocess data
# ----------------------------
df = pd.read_excel("LIDC-IDRI nodule metadata.xlsx")
df = df[df['malignancy'] != 3]
df = df[df['nodule selected'] != 'No'] # 955_samples
df['label'] = df['malignancy'].apply(lambda x: 1 if x >= 4 else 0)

df[['centroid_x', 'centroid_y', 'centroid_z']] = df['nodule centroid'].apply(
    lambda x: pd.Series(eval(x) if isinstance(x, str) else x)
)


# counting anatomic position_________
unique_positions = df['anatomic position'].unique()
num_unique = len(unique_positions)
print(f"Number of unique anatomic positions: {num_unique}")
print("Unique values:", unique_positions)


# Define features
categorical_features = ['subtlety', 'internal structure', 'calcification', 'sphericity',
                        'margin', 'lobulation', 'spiculation', 'textures']
numerical_features = ['nodule diameter (mm)', 'centroid_x', 'centroid_y', 'centroid_z']
feature_cols = numerical_features + categorical_features

X = df[feature_cols]
y = df['label'].astype(int)

# ----------------------------
# Preprocessing
# ----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# ----------------------------
# Define models
# ----------------------------
base_models = {
    'SVM': SVC(probability=True, kernel='rbf', random_state=63),
    'kNN': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=63),
    'DecisionTree': DecisionTreeClassifier(random_state=63),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=63),
    'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=63),
    'AdaBoost': AdaBoostClassifier(random_state=63)
}

# Build pipelines
models = {
    name: Pipeline(steps=[('pre', preprocessor), ('clf', model)])
    for name, model in base_models.items()
}

# ----------------------------
# 5-fold cross-validation
# ----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=63)
results = []
roc_data = {}

for name, pipeline in models.items():
    accs, precs, recs, f1s, specs, senss, aucs = [], [], [], [], [], [], []
    all_y_val, all_y_proba = [], []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]

        cm = confusion_matrix(y_val, y_pred)
        TN, FP, FN, TP = cm.ravel()

        accs.append(accuracy_score(y_val, y_pred))
        precs.append(precision_score(y_val, y_pred))
        recs.append(recall_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred))
        specs.append(TN / (TN + FP))
        senss.append(TP / (TP + FN))
        aucs.append(roc_auc_score(y_val, y_proba))

        all_y_val.extend(y_val)
        all_y_proba.extend(y_proba)

    # Store mean ± std metrics
    results.append({
        'Model': name,
        'Accuracy': f"{np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}",
        'Precision': f"{np.mean(precs)*100:.2f} ± {np.std(precs)*100:.2f}",
        'Recall (Sensitivity)': f"{np.mean(recs)*100:.2f} ± {np.std(recs)*100:.2f}",
        'Specificity': f"{np.mean(specs)*100:.2f} ± {np.std(specs)*100:.2f}",
        'F1 Score': f"{np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}",
        'AUROC': f"{np.mean(aucs)*100:.2f} ± {np.std(aucs)*100:.2f}"
    })

    fpr, tpr, _ = roc_curve(all_y_val, all_y_proba)
    overall_auc = roc_auc_score(all_y_val, all_y_proba)
    roc_data[name] = (fpr, tpr, overall_auc)


with open(f"{code_outputs}/confusion_matrices_all_models.txt", "w") as f:
    for name, pipeline in models.items():
        preds = cross_val_predict(pipeline, X, y, cv=cv, method='predict')
        cm = confusion_matrix(y, preds)
        TN, FP, FN, TP = cm.ravel()
        f.write(f"Model: {name}\n")
        f.write(f"Confusion Matrix:\n[[{TN} {FP}]\n [{FN} {TP}]]\n")
        f.write(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}\n")
        f.write("-" * 50 + "\n")



# ----------------------------
# Plot ROC curves
# ----------------------------
plt.figure(figsize=(10, 7))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves for All Models", fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{code_outputs}/roc_comparison_all_models.png", dpi=300)
plt.show()



# ----------------------------
# Save model comparison table
# ----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(f"{code_outputs}/model_comparison_results.csv", index=False, encoding='utf-8-sig')
print(results_df)




# ----------------------------
# Learning Curves (Accuracy)
# ----------------------------
train_sizes = np.linspace(0.1, 0.9, 9)
X_train_full, X_test_holdout, y_train_full, y_test_holdout = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=63
)

plt.figure(figsize=(7, 4))

for name, pipeline in models.items():
    train_acc, test_acc = [], []
    train_loss, test_loss = [], []

    for frac in train_sizes:
        X_part, _, y_part, _ = train_test_split(X_train_full, y_train_full, train_size=frac, stratify=y_train_full)
        model_clone = clone(pipeline)
        model_clone.fit(X_part, y_part)

        y_part_pred = model_clone.predict(X_part)
        y_test_pred = model_clone.predict(X_test_holdout)

        train_acc.append(accuracy_score(y_part, y_part_pred))
        test_acc.append(accuracy_score(y_test_holdout, y_test_pred))

        if hasattr(model_clone.named_steps['clf'], "predict_proba"):
            y_part_proba = model_clone.predict_proba(X_part)
            y_test_proba = model_clone.predict_proba(X_test_holdout)
            train_loss.append(log_loss(y_part, y_part_proba))
            test_loss.append(log_loss(y_test_holdout, y_test_proba))
        else:
            train_loss.append(np.nan)
            test_loss.append(np.nan)

    pd.DataFrame({
        "Train_Size": train_sizes,
        "Train_Accuracy": train_acc,
        "Test_Accuracy": test_acc,
        "Train_Loss": train_loss,
        "Test_Loss": test_loss
    }).to_csv(f"{code_outputs}/{name}_learning_curve.csv", index=False, encoding='utf-8-sig')

    plt.plot(train_sizes, test_acc, label=name, linewidth=1)

plt.xlabel("Training Size Fraction", fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
# plt.title("Learning Curves (Test Accuracy)", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.9, linewidth=0.5)
plt.tight_layout()
plt.savefig(f"{code_outputs}/learning_curves_accuracy.png", dpi=300)
plt.show()


# Add another section for making smooth curve



#%%
# ----------------------------
# Learning Curves with Smooth Curves
# ----------------------------
from scipy.interpolate import interp1d

# Define distinct colors for each model
colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']
color_map = {name: colors[i] for i, name in enumerate(models.keys())}

plt.figure(figsize=(7, 3.5))

for name, pipeline in models.items():
    train_acc, test_acc = [], []
    train_loss, test_loss = [], []

    for frac in train_sizes:
        X_part, _, y_part, _ = train_test_split(X_train_full, y_train_full, train_size=frac, stratify=y_train_full)
        model_clone = clone(pipeline)
        model_clone.fit(X_part, y_part)

        y_part_pred = model_clone.predict(X_part)
        y_test_pred = model_clone.predict(X_test_holdout)

        train_acc.append(accuracy_score(y_part, y_part_pred))
        test_acc.append(accuracy_score(y_test_holdout, y_test_pred))

        if hasattr(model_clone.named_steps['clf'], "predict_proba"):
            y_part_proba = model_clone.predict_proba(X_part)
            y_test_proba = model_clone.predict_proba(X_test_holdout)
            train_loss.append(log_loss(y_part, y_part_proba))
            test_loss.append(log_loss(y_test_holdout, y_test_proba))
        else:
            train_loss.append(np.nan)
            test_loss.append(np.nan)

    # Create smooth curves using interpolation
    # Generate more points for smooth interpolation (reduced from 100 to 50 for less smoothing)
    smooth_x = np.linspace(train_sizes.min(), train_sizes.max(), 100)
    
    # Interpolate test accuracy for smooth curve (using linear instead of cubic for less smoothing)
    if len(test_acc) > 1:  # Need at least 2 points for interpolation
        f_test = interp1d(train_sizes, test_acc, kind='linear', bounds_error=False, fill_value='extrapolate')
        smooth_test_acc = f_test(smooth_x)
        
        # Plot original points and smooth curve
        plt.plot(train_sizes, test_acc, 'o', markersize=2, alpha=0.7, color=color_map[name])  # Original points
        plt.plot(smooth_x, smooth_test_acc, label=name, linewidth=1, alpha=0.8, color=color_map[name])  # Smooth curve
    else:
        plt.plot(train_sizes, test_acc, 'o-', label=name, linewidth=1, color=color_map[name])

plt.xlabel("Training Size Fraction", fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
# plt.title("Learning Curves with Smooth Interpolation", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.9, linewidth=0.5)
plt.tight_layout()
plt.savefig(f"{code_outputs}/learning_curves_smooth_accuracy.png", dpi=300)
plt.show()
#%%



# ----------------------------
# Learning Curves (Train vs Test Accuracy)
# ----------------------------
import matplotlib.cm as cm

# colors = cm.get_cmap('tab10', len(models))
# model_color_map = {name: colors(i) for i, name in enumerate(models)}

colors = plt.colormaps.get_cmap('tab10')  # or matplotlib.colormaps['tab10']
model_color_map = {name: colors(i / len(models)) for i, name in enumerate(models)}



plt.figure(figsize=(12, 6))

for name, pipeline in models.items():
    train_acc, test_acc = [], []

    for frac in train_sizes:
        X_part, _, y_part, _ = train_test_split(X_train_full, y_train_full, train_size=frac, stratify=y_train_full)
        model_clone = clone(pipeline)
        model_clone.fit(X_part, y_part)

        y_part_pred = model_clone.predict(X_part)
        y_test_pred = model_clone.predict(X_test_holdout)

        train_acc.append(accuracy_score(y_part, y_part_pred))
        test_acc.append(accuracy_score(y_test_holdout, y_test_pred))

    pd.DataFrame({
        "Train_Size": train_sizes,
        "Train_Accuracy": train_acc,
        "Test_Accuracy": test_acc
    }).to_csv(f"{code_outputs}/{name}_learning_curve2.csv", index=False, encoding='utf-8-sig')

    plt.plot(train_sizes, train_acc, linestyle='--', color=model_color_map[name], label=f"{name} (Train)", linewidth=1.5)
    plt.plot(train_sizes, test_acc, linestyle='-', color=model_color_map[name], label=f"{name} (Test)", linewidth=2)

plt.xlabel("Training Size Fraction", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Learning Curves (Train vs Test Accuracy)", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{code_outputs}/learning_curves_train_test_accuracy.png", dpi=300)
plt.show()



# ----------------------------
# SHAP for Best Model (Random Forest)
# ----------------------------
import shap
from sklearn.ensemble import RandomForestClassifier

# Transform data and convert to dense if needed
X_transformed = preprocessor.fit_transform(X)
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

feature_names = preprocessor.get_feature_names_out()

# Fit Random Forest
best_model = RandomForestClassifier(n_estimators=100, random_state=63)
best_model.fit(X_transformed, y)

# TreeExplainer for Random Forest
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_transformed)

# Class 1 SHAP values (binary classification)
shap_vals = shap_values[:, :, 1]  # class 1


# Create Explanation for class 1
shap_expl = shap.Explanation(
    values=shap_vals,
    base_values=explainer.expected_value[1],
    data=X_transformed,
    feature_names=feature_names
)

# Now use shap.plots.beeswarm
plt.figure(figsize=(10, 7))
shap.plots.beeswarm(shap_expl, max_display=20, show=False)
plt.title("SHAP Beeswarm (Random Forest)")
plt.tight_layout()
plt.savefig(f"{code_outputs}/shap_beeswarm_rf.png", dpi=300, bbox_inches='tight')
plt.show()

# Bar Plot
plt.figure(figsize=(10, 7))
shap.plots.bar(shap_expl, max_display=20, show=False)
plt.title("SHAP Bar Importance (Random Forest)")
plt.tight_layout()
plt.savefig(f"{code_outputs}/shap_bar_importance_rf.png", dpi=300, bbox_inches='tight')
plt.show()

# Waterfall Plot (Sample 0)
plt.figure(figsize=(10, 7))
shap.plots.waterfall(shap_expl[0], max_display=20, show=False)
plt.title("SHAP Waterfall Plot (Sample 0)")
plt.tight_layout()
plt.savefig(f"{code_outputs}/shap_waterfall_sample0_rf.png", dpi=300, bbox_inches='tight')
plt.show()


# Extra: Not needed Here
# ----------------------------
# SHAP for Best Model (XGBoost) 
# ----------------------------
# import shap

# # Transform data and convert to dense
# X_transformed = preprocessor.fit_transform(X)
# if hasattr(X_transformed, "toarray"):
#     X_transformed = X_transformed.toarray()

# feature_names = preprocessor.get_feature_names_out()

# # Fit best model
# best_model = xgb.XGBClassifier(eval_metric='logloss', random_state=63)
# best_model.fit(X_transformed, y)

# # SHAP Explainer
# explainer = shap.Explainer(best_model, X_transformed, feature_names=feature_names)
# shap_values = explainer(X_transformed)

# # Beeswarm Plot
# plt.figure(figsize=(10, 7))
# shap.plots.beeswarm(shap_values, max_display=20, show=False)
# plt.title("SHAP Beeswarm (XGBoost)")
# plt.tight_layout()
# plt.savefig(f"{code_outputs}/shap_beeswarm.png", dpi=300, bbox_inches='tight')
# plt.show()

# # Bar Plot
# plt.figure(figsize=(10, 7))
# shap.plots.bar(shap_values, max_display=20, show=False)
# plt.title("SHAP Bar Importance (XGBoost)")
# plt.tight_layout()
# plt.savefig(f"{code_outputs}/shap_bar_importance.png", dpi=300, bbox_inches='tight')
# plt.show()

# # Waterfall Plot (Sample 0)
# plt.figure(figsize=(10, 7))
# shap.plots.waterfall(shap_values[0], max_display=20, show=False)
# plt.title("SHAP Waterfall Plot (Sample 0)")
# plt.tight_layout()
# plt.savefig(f"{code_outputs}/shap_waterfall_sample0.png", dpi=300, bbox_inches='tight')
# plt.show()
