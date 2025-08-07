import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel
import torch
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import html
import os

code_outputs = 'code outputs/c30 PubMedBERT captum'
os.makedirs(code_outputs, exist_ok=True)
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
text = texts[0]
# Tokenize input text
text_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, padding=True, max_length=512, return_tensors='pt')
# Fix: convert tensor to list of IDs for token conversion
print(tokenizer.convert_ids_to_tokens(text_ids[0].tolist()))




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


# ================== Captum Integrated Gradients Visualization =====================
# For Captum, we need a differentiable output. Since model_clf is a scikit-learn model (not differentiable),
# we will attribute with respect to the mean pooled embedding's first dimension as a demonstration.
# For true class probability attribution, you would need a differentiable classifier head (e.g., torch.nn.Linear).

def attribute_sample(text, tokenizer, model):
    encoded = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    # Get embeddings for input_ids
    embedding_layer = model.get_input_embeddings()
    input_embeds = embedding_layer(input_ids)

    def forward_func(embeds, attention_mask):
        # Pass embeddings directly to the model
        outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return pooled[:, 0]  # attribute to first neuron

    ig = IntegratedGradients(forward_func)
    attributions, delta = ig.attribute(inputs=input_embeds, additional_forward_args=(attention_mask,), return_convergence_delta=True)
    return attributions, input_ids, delta

# Function to merge subwords and average their attributions for visualization

def merge_subwords(tokens, attributions):
    words = []
    word_attributions = []
    current_word = ""
    current_attr = 0.0
    count = 0
    for token, attr in zip(tokens, attributions):
        if token.startswith("##") and words:
            current_word += token[2:]
            current_attr += attr
            count += 1
        else:
            if current_word:
                words.append(current_word)
                word_attributions.append(current_attr / count)
            current_word = token
            current_attr = attr
            count = 1
    if current_word:
        words.append(current_word)
        word_attributions.append(current_attr / count)
    return words, word_attributions

# Select 10 random unique indices from the dataset for visualization
np.random.seed(RANDOM_STATE_RABIUL)  # For reproducibility
random_indices = np.random.choice(len(texts), size=10, replace=False)

html_outputs = []
for idx in random_indices:
    text = texts[idx]
    attributions, input_ids, delta = attribute_sample(text, tokenizer, model)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # # Sum attributions across embedding dimensions for each token
    attr = attributions[0].detach().numpy().sum(axis=1)
    # # Normalize by max absolute value, preserving sign (so both green and red are visible)
    attr = attr / (np.abs(attr).max() + 1e-10)
    # Invert attributions so red indicates malignant (positive class) and green indicates benign (negative class)
    # attr = -attr


    
    # Merge subwords for visualization
    words, word_attr = merge_subwords(tokens, attr)
    # Remove [CLS] and [SEP] tokens and their attributions
    filtered = [(w, a) for w, a in zip(words, word_attr) if w not in ("[CLS]", "[SEP]")]
    if filtered:
        words, word_attr = zip(*filtered)
    else:
        words, word_attr = [], []
    # Get the true label for this sample
    true_label = int(y[idx])
    
    # Get the embedding for this sample for prediction
    with torch.no_grad():
        sample_embed = model(**tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)).last_hidden_state.mean(dim=1).numpy()
    pred_proba = float(model_clf.predict_proba(sample_embed)[0, 1])
    pred_class = int(model_clf.predict(sample_embed)[0])
    vis_html = viz.visualize_text([
        viz.VisualizationDataRecord(
            word_attributions=word_attr,
            pred_prob=pred_proba,
            pred_class=pred_class,
            true_class=true_label,
            attr_class='malignant',  # Label for attribution class - red will be associated with malignant
            attr_score=delta.item(),
            raw_input_ids=words,
            convergence_score=delta.item()
        )
    ])
    html_outputs.append(vis_html.data)

with open(f"{code_outputs}/attribution_output.html", "w", encoding="utf-8") as f:
    for html_str in html_outputs:
        f.write(html_str)
        f.write("<hr>")  # separator between samples




