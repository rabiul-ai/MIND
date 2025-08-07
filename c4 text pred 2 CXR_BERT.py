import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel


code_outputs = 'code outputs/c4 text pred 2 CXR_BERT'
n_nodule = 955 
n_nodule = 1385 # make one line comment
RANDOM_STATE_RABIUL = 63


# Load data
df = pd.read_excel("LIDC-IDRI nodule metadata.xlsx")
df = df[df['malignancy'] != 3]
if n_nodule == 955:
    df = df[df['nodule selected'] != 'No'] # keep only the eligible nodules
df['label'] = df['malignancy'].apply(lambda x: 1 if x >= 4 else 0)



# Load CXR-BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    'microsoft/BiomedVLP-CXR-BERT-specialized',
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    'microsoft/BiomedVLP-CXR-BERT-specialized',
    trust_remote_code=True
)
model.eval()  # Disable dropout etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Encode text (nodule descriptions)
texts = df['nodule description'].tolist()
encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)
encoded = {key: val.to(device) for key, val in encoded.items()}

# Get embeddings via mean pooling
with torch.no_grad():
    output = model(**encoded)
    embeddings = output.last_hidden_state.mean(dim=1).cpu().numpy()

X = embeddings
y = df['label'].values

# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE_RABIUL)
accs, precs, recs, f1s, specs, senss, aucs = [], [], [], [], [], [], []

for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]

    cm = confusion_matrix(y_val, y_pred)
    TN, FP, FN, TP = cm.ravel()

    accs.append(accuracy_score(y_val, y_pred))
    precs.append(precision_score(y_val, y_pred))
    recs.append(recall_score(y_val, y_pred))
    f1s.append(f1_score(y_val, y_pred))
    specs.append(TN / (TN + FP))
    senss.append(TP / (TP + FN))
    aucs.append(roc_auc_score(y_val, y_proba))

# Save results
results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1 Score", "AUROC"],
    "Mean ± Std": [
        f"{np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}",
        f"{np.mean(precs)*100:.2f} ± {np.std(precs)*100:.2f}",
        f"{np.mean(recs)*100:.2f} ± {np.std(recs)*100:.2f}",
        f"{np.mean(specs)*100:.2f} ± {np.std(specs)*100:.2f}",
        f"{np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}",
        f"{np.mean(aucs)*100:.2f} ± {np.std(aucs)*100:.2f}",
    ]
})


# Save results as columns
results = pd.DataFrame({
    "Accuracy": [f"{np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}"],
    "Precision": [f"{np.mean(precs)*100:.2f} ± {np.std(precs)*100:.2f}"],
    "Recall (Sensitivity)": [f"{np.mean(recs)*100:.2f} ± {np.std(recs)*100:.2f}"],
    "Specificity": [f"{np.mean(specs)*100:.2f} ± {np.std(specs)*100:.2f}"],
    "F1 Score": [f"{np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}"],
    "AUROC": [f"{np.mean(aucs)*100:.2f} ± {np.std(aucs)*100:.2f}"]
})


results.to_csv(f"{code_outputs}/cxr_bert_prediction_{n_nodule}.csv", index=False, encoding="utf-8-sig")
print(results)
