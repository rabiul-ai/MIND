import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel
import torch

code_outputs = 'code outputs/c12 PubMedBERT pred saving'
n_nodule = 955
n_nodule = 1385 # 1385
RANDOM_STATE_RABIUL = 63

# Load data
df = pd.read_excel("LIDC-IDRI nodule metadata.xlsx")
df = df[df['malignancy'] != 3]
if n_nodule == 955:
    df = df[df['nodule selected'] != 'No'] # 955_samples
df['label'] = df['malignancy'].apply(lambda x: 1 if x >= 4 else 0)


df2 = df[df['why not selected'] == 'cube shape not 32']
df['malignancy'].value_counts().sort_index()

# Tokenize and embed with PubMedBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")


texts = df['nodule description'].tolist()
encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')

with torch.no_grad():
    output = model(**encoded)
    embeddings = output.last_hidden_state.mean(dim=1).numpy()

# Prepare labels
y = df['label'].values
nodule_ids = df['nodule ID'].to_numpy()
nodule_ids = [str(i) for i in nodule_ids]

# Cross-validation
metrics = {'Accuracy': [], 'Precision': [], 'Recall (Sensitivity)': [], 'Specificity': [], 'F1 Score': [], 'AUROC': []}


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE_RABIUL)
folds = list(kf.split(nodule_ids, y))

'''_________________Opening New Dictionary for Saving Decision______________'''
Text_score = {}
keys = ['y_true_fold','y_pred_fold', 'y_proba_fold', 'y_true_all','y_pred_all', 'y_proba_all']
for key in keys:
    Text_score[key] = []

# nodule embedding
nodule_embeddings = {}

all_test_nodules = []
# for i in range(len(all_test_nodules)):
#     print(len(set(all_test_nodules[i])))

# for train_idx, val_idx in cv.split(embeddings, y):

for fold, (train_idx, val_idx) in enumerate(folds):
    X_train, X_val = embeddings[train_idx], embeddings[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    """==================== Extracting Embedding 768 ==================="""
    # making embedding dict per nid
    test_nodules = [nodule_ids[i] for i in val_idx]
    all_test_nodules.append(test_nodules)
    # print(len(test_nodules))
    # test_nodules = nodule_ids[val_idx]
    for i, nid in enumerate(test_nodules):
        # print(i, len(nodule_embeddings))
        nodule_embeddings[nid] = X_val[i] # 768
    #---------------------------------------------------------------------------

    model_clf = LogisticRegression(max_iter=1000)
    model_clf.fit(X_train, y_train)
    y_pred = model_clf.predict(X_val)
    y_proba = model_clf.predict_proba(X_val)[:, 1]
    
    
    # Making dict __________________________________________________
    y_test = y_val
    y_pred_prob = y_proba

    Text_score['y_true_fold'].append([int(x) for x in y_test])
    Text_score['y_pred_fold'].append([int(x) for x in y_pred])
    Text_score['y_proba_fold'].append([round(float(x), 5) for x in y_pred_prob])

    Text_score['y_true_all'].extend([int(x) for x in y_test])
    Text_score['y_pred_all'].extend([int(x) for x in y_pred])
    Text_score['y_proba_all'].extend([round(float(x), 5) for x in y_pred_prob])
    #_________________________________________________________________

    cm = confusion_matrix(y_val, y_pred)
    TN, FP, FN, TP = cm.ravel()

    metrics['Accuracy'].append(accuracy_score(y_val, y_pred))
    metrics['Precision'].append(precision_score(y_val, y_pred))
    metrics['Recall (Sensitivity)'].append(recall_score(y_val, y_pred))
    metrics['Specificity'].append(TN / (TN + FP))
    metrics['F1 Score'].append(f1_score(y_val, y_pred))
    metrics['AUROC'].append(roc_auc_score(y_val, y_proba))


"""=================== Saving Decision ================================="""
import pickle
# saving pickle file_____________________
with open(f'{code_outputs}/Text_score_PubMedBERT_{n_nodule}.pkl', 'wb') as file:
    pickle.dump(Text_score, file)
    
    
"""=============================Saving Embeddings=============================="""
# saving pickle file_____________________
with open(f'{code_outputs}/PubMedBERT_nodule_embeddings_{n_nodule}.pkl', 'wb') as file:
    pickle.dump(nodule_embeddings, file)


# Create results DataFrame
results_df = pd.DataFrame({
    metric: [f"{np.mean(scores)*100:.2f} ± {np.std(scores)*100:.2f}"]
    for metric, scores in metrics.items()
})

# Save to CSV
results_df.to_csv(f"{code_outputs}/PubMedBERT_prediction_{n_nodule}.csv", index=False, encoding='utf-8-sig')

# Display
print(results_df)
