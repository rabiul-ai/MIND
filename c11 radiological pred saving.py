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
import pickle

code_outputs = 'code outputs/c11 radiological pred saving'
n_nodule = '955' 
# n_nodule = '1385'
RANDOM_STATE_RABIUL = 63

# ----------------------------
# Load and preprocess data
# ----------------------------
df = pd.read_excel("LIDC-IDRI nodule metadata.xlsx")
df = df[df['malignancy'] != 3]
if n_nodule == '955':
    df = df[df['nodule selected'] != 'No'] # 955_samples
df['label'] = df['malignancy'].apply(lambda x: 1 if x >= 4 else 0)

df[['centroid_x', 'centroid_y', 'centroid_z']] = df['nodule centroid'].apply(
    lambda x: pd.Series(eval(x) if isinstance(x, str) else x)
)

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
# Saving Radiological Fetures as Embedding
# -----------------------------
# Create dictionary: key = nodule ID, value = numpy array of features (float)
# Normalize features before saving embedding
preprocessor.fit(X)
features_norm = preprocessor.transform(X)
if hasattr(features_norm, 'toarray'):
    features_norm = features_norm.toarray()
nodule_ids = df['nodule ID'].astype(str).tolist()
rad_embedding_dict = {nid: features_norm[i] for i, nid in enumerate(nodule_ids)}

# Save to pickle file
with open(f'{code_outputs}/Radiological_embedding_{n_nodule}.pkl', 'wb') as f:
    pickle.dump(rad_embedding_dict, f)


# ----------------------------
# Define models
# ----------------------------
base_models = {
    'SVM': SVC(probability=True, kernel='rbf', random_state=RANDOM_STATE_RABIUL)
}

# Build pipelines
models = {
    name: Pipeline(steps=[('pre', preprocessor), ('clf', model)])
    for name, model in base_models.items()
}

# ----------------------------
# 5-fold cross-validation
# ----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE_RABIUL)
results = []
roc_data = {}

for name, pipeline in models.items():
    accs, precs, recs, f1s, specs, senss, aucs = [], [], [], [], [], [], []
    all_y_val, all_y_proba = [], []
    
    
    '''_________________Opening New Dictionary for Saving Decision______________'''
    Radiological_score = {}
    keys = ['y_true_fold','y_pred_fold', 'y_proba_fold', 'y_true_all','y_pred_all', 'y_proba_all']
    for key in keys:
        Radiological_score[key] = []
    

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
        
        # Making dict __________________________________________________
        y_test = y_val
        y_pred_prob = y_proba
        
        Radiological_score['y_true_fold'].append(list(y_test))
        Radiological_score['y_pred_fold'].append([int(x) for x in y_pred])
        Radiological_score['y_proba_fold'].append([round(float(x), 5) for x in y_pred_prob])

        Radiological_score['y_true_all'].extend(list(y_test))
        Radiological_score['y_pred_all'].extend([int(x) for x in y_pred])
        Radiological_score['y_proba_all'].extend([round(float(x), 5) for x in y_pred_prob])

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
    
    
    """=================== Saving Decision ================================="""
    import pickle
    # saving pickle file_____________________
    with open(f'{code_outputs}/Radiological_score_SVM_{n_nodule}.pkl', 'wb') as file:
        pickle.dump(Radiological_score, file)



with open(f"{code_outputs}/confusion_matrices_all_models_{n_nodule}.txt", "w") as f:
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
plt.savefig(f"{code_outputs}/roc_comparison_all_models_{n_nodule}.png", dpi=300)
plt.show()



# ----------------------------
# Save model comparison table
# ----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(f"{code_outputs}/model_comparison_results_{n_nodule}.csv", index=False, encoding='utf-8-sig')
print(results_df)


