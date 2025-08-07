import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, log_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import clone
import xgboost as xgb
import matplotlib.pyplot as plt


code_outputs = 'code outputs/c2 radiological feature performance_955_samples'


# ----------------------------
# Load and preprocess data
# ----------------------------
df = pd.read_excel("LIDC-IDRI nodule metadata.xlsx")
df = df[df['malignancy'] != 3]
df = df[df['nodule selected'] != 'No'] # keep only the eligible nodules
df['label'] = df['malignancy'].apply(lambda x: 1 if x >= 4 else 0)

df[['centroid_x', 'centroid_y', 'centroid_z']] = df['nodule centroid'].apply(
    lambda x: pd.Series(eval(x) if isinstance(x, str) else x)
)

feature_cols = [
    'nodule diameter (mm)', 'centroid_x', 'centroid_y', 'centroid_z',
    'subtlety', 'internal structure', 'calcification', 'sphericity',
    'margin', 'lobulation', 'spiculation', 'textures'
]
X = df[feature_cols].astype(float)
y = df['label'].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# Define models
# ----------------------------
models = {
    'SVM': SVC(probability=True, kernel='rbf', random_state=42),
    'kNN': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}


# ----------------------------
# 5 fold cv result
# ----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
roc_data = {}  # Collect ROC data for plotting

for name, model in models.items():
    accs, precs, recs, f1s, specs, senss, aucs = [], [], [], [], [], [], []

    all_y_val = []
    all_y_proba = []

    for train_idx, val_idx in cv.split(X_scaled, y):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        cloned_model = clone(model)
        cloned_model.fit(X_train, y_train)

        y_pred = cloned_model.predict(X_val)
        y_proba = cloned_model.predict_proba(X_val)[:, 1]

        cm = confusion_matrix(y_val, y_pred)
        TN, FP, FN, TP = cm.ravel()

        accs.append(accuracy_score(y_val, y_pred))
        precs.append(precision_score(y_val, y_pred))
        recs.append(recall_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred))
        specs.append(TN / (TN + FP))
        senss.append(TP / (TP + FN))
        aucs.append(roc_auc_score(y_val, y_proba))

        # Collect for ROC
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

    # Compute ROC curve for the whole set
    fpr, tpr, _ = roc_curve(all_y_val, all_y_proba)
    overall_auc = roc_auc_score(all_y_val, all_y_proba)
    roc_data[name] = (fpr, tpr, overall_auc)


# Save confusion matrices
with open(f"{code_outputs}/confusion_matrices_all_models.txt", "w") as f:
    for name, model in models.items():
        preds = cross_val_predict(clone(model), X_scaled, y, cv=cv, method='predict')
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
# Learning Curves (Accuracy & Loss)
# ----------------------------
train_sizes = np.linspace(0.1, 0.9, 9)
X_train_full, X_test_holdout, y_train_full, y_test_holdout = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

plt.figure(figsize=(12, 6))

for name, model in models.items():
    train_acc, test_acc = [], []
    train_loss, test_loss = [], []

    for frac in train_sizes:
        X_part, _, y_part, _ = train_test_split(X_train_full, y_train_full, train_size=frac, stratify=y_train_full)
        cloned_model = clone(model)
        cloned_model.fit(X_part, y_part)

        y_part_pred = cloned_model.predict(X_part)
        y_test_pred = cloned_model.predict(X_test_holdout)

        train_acc.append(accuracy_score(y_part, y_part_pred))
        test_acc.append(accuracy_score(y_test_holdout, y_test_pred))

        if hasattr(cloned_model, "predict_proba"):
            y_part_proba = cloned_model.predict_proba(X_part)
            y_test_proba = cloned_model.predict_proba(X_test_holdout)
            train_loss.append(log_loss(y_part, y_part_proba))
            test_loss.append(log_loss(y_test_holdout, y_test_proba))
        else:
            train_loss.append(np.nan)
            test_loss.append(np.nan)

    # Save learning curve data
    pd.DataFrame({
        "Train_Size": train_sizes,
        "Train_Accuracy": train_acc,
        "Test_Accuracy": test_acc,
        "Train_Loss": train_loss,
        "Test_Loss": test_loss
    }).to_csv(f"{code_outputs}/{name}_learning_curve.csv", index=False, encoding='utf-8-sig')

    # Plot test accuracy
    plt.plot(train_sizes, test_acc, label=name, linewidth=2)

plt.xlabel("Training Size Fraction", fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
plt.title("Learning Curves (Test Accuracy)", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{code_outputs}/learning_curves_accuracy.png", dpi=300)
plt.show()


# ----------------------------
# Learning Curves (Accuracy & Loss). Second graph
# ----------------------------
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split


# Generate a unique color for each model
model_names = list(models.keys())
colors = plt.colormaps.get_cmap('tab10')
model_color_map = {name: colors(i / len(model_names)) for i, name in enumerate(model_names)}

# Plotting
plt.figure(figsize=(12, 6))

for i, (name, model) in enumerate(models.items()):
    train_acc, test_acc = [], []
    train_loss, test_loss = [], []

    for frac in train_sizes:
        X_part, _, y_part, _ = train_test_split(X_train_full, y_train_full, train_size=frac, stratify=y_train_full)
        cloned_model = clone(model)
        cloned_model.fit(X_part, y_part)

        y_part_pred = cloned_model.predict(X_part)
        y_test_pred = cloned_model.predict(X_test_holdout)

        train_acc.append(accuracy_score(y_part, y_part_pred))
        test_acc.append(accuracy_score(y_test_holdout, y_test_pred))

        if hasattr(cloned_model, "predict_proba"):
            y_part_proba = cloned_model.predict_proba(X_part)
            y_test_proba = cloned_model.predict_proba(X_test_holdout)
            train_loss.append(log_loss(y_part, y_part_proba))
            test_loss.append(log_loss(y_test_holdout, y_test_proba))
        else:
            train_loss.append(np.nan)
            test_loss.append(np.nan)

    # Save learning curve data
    pd.DataFrame({
        "Train_Size": train_sizes,
        "Train_Accuracy": train_acc,
        "Test_Accuracy": test_acc,
        "Train_Loss": train_loss,
        "Test_Loss": test_loss
    }).to_csv(f"{code_outputs}/{name}_learning_curve2.csv", index=False, encoding='utf-8-sig')

    # Plot curves with consistent color
    plt.plot(train_sizes, train_acc, linestyle='--', color=model_color_map[name], label=f"{name} (Train)", linewidth=1.5)
    plt.plot(train_sizes, test_acc, linestyle='-', color=model_color_map[name], label=f"{name} (Test)", linewidth=2)

# Plot formatting
plt.xlabel("Training Size Fraction", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Learning Curves (Train vs Test Accuracy)", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{code_outputs}/learning_curves_train_test_accuracy.png", dpi=300)
plt.show()




# ----------------------------
# SHAP graphs
# ----------------------------
import shap

# Use best-performing model for SHAP
best_model_name = "XGBoost"
best_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
best_model.fit(X_scaled, y)

# Create SHAP explainer with proper feature names
explainer = shap.Explainer(best_model, X_scaled, feature_names=feature_cols)
shap_values = explainer(X_scaled)

# SHAP Beeswarm Plot
plt.figure(figsize=(10, 7))  # Adjust size if needed
shap.plots.beeswarm(shap_values, max_display=12, show=False)
plt.title(f"{best_model_name} - SHAP Beeswarm Plot")
plt.tight_layout()
plt.savefig(f"{code_outputs}/shap_beeswarm.png", dpi=300, bbox_inches='tight')
plt.show()

# Create a new figure to avoid cropping
plt.figure(figsize=(10, 7))
shap.plots.bar(shap_values, max_display=12, show=False)
plt.title(f"{best_model_name} - SHAP Feature Importance (Bar)")
plt.tight_layout()  # Make sure layout fits
plt.savefig(f"{code_outputs}/shap_bar_importance.png", dpi=300, bbox_inches='tight')
plt.show()

# SHAP Waterfall Plot for one instance
plt.figure(figsize=(10, 7))  # Adjust size if needed
shap.plots.waterfall(shap_values[0], max_display=12, show=False)
plt.title(f"{best_model_name} - SHAP Waterfall (Sample 0)")
plt.tight_layout()
plt.savefig(f"{code_outputs}/shap_waterfall_sample0.png", dpi=300, bbox_inches='tight')
plt.show()
