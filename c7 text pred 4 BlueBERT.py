import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel
import torch

code_outputs = 'code outputs/c7 text pred 4 BlueBERT'

n_nodule = 955 
n_nodule = 1385 # make one line comment
RANDOM_STATE_RABIUL = 63


# Load data
df = pd.read_excel("LIDC-IDRI nodule metadata.xlsx")
df = df[df['malignancy'] != 3]
if n_nodule == 955:
    df = df[df['nodule selected'] != 'No'] # keep only the eligible nodules
df['label'] = df['malignancy'].apply(lambda x: 1 if x >= 4 else 0)

# Tokenize and embed with BlueBERT
tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
model = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")


texts = df['nodule description'].tolist()
encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')

with torch.no_grad():
    output = model(**encoded)
    embeddings = output.last_hidden_state.mean(dim=1).numpy()

# Prepare labels
y = df['label'].values

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE_RABIUL)
metrics = {'Accuracy': [], 'Precision': [], 'Recall (Sensitivity)': [], 'Specificity': [], 'F1 Score': [], 'AUROC': []}

for train_idx, val_idx in cv.split(embeddings, y):
    X_train, X_val = embeddings[train_idx], embeddings[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model_clf = LogisticRegression(max_iter=1000)
    model_clf.fit(X_train, y_train)
    y_pred = model_clf.predict(X_val)
    y_proba = model_clf.predict_proba(X_val)[:, 1]

    cm = confusion_matrix(y_val, y_pred)
    TN, FP, FN, TP = cm.ravel()

    metrics['Accuracy'].append(accuracy_score(y_val, y_pred))
    metrics['Precision'].append(precision_score(y_val, y_pred))
    metrics['Recall (Sensitivity)'].append(recall_score(y_val, y_pred))
    metrics['Specificity'].append(TN / (TN + FP))
    metrics['F1 Score'].append(f1_score(y_val, y_pred))
    metrics['AUROC'].append(roc_auc_score(y_val, y_proba))

# Create results DataFrame
results_df = pd.DataFrame({
    metric: [f"{np.mean(scores)*100:.2f} ± {np.std(scores)*100:.2f}"]
    for metric, scores in metrics.items()
})

# Save to CSV
results_df.to_csv(f"{code_outputs}/BlueBERT_prediction_{n_nodule}.csv", index=False, encoding='utf-8-sig')

# Display
print(results_df)
