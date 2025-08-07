'''
Date: July 11, 2025
Task: Image, Radiological, TXT Decision Fusion 
AVERAGE DECISION FUSION 
'''

import pickle, os
import numpy as np
from sklearn.metrics import f1_score
from skopt import gp_minimize
from skopt.space import Real
from scipy.stats import gaussian_kde
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd

n_nodule = 955
# n_nodule = 1385
modality_type = 'IMG_RAD_TXT'


"""==============Step 1: Searching Weights for Fusion of Image, Radiological and Text ============"""
"""______________________ Loading Scores ____________________________"""
# input file directories
score_folder = f'code inputs/c13 img rad text predictions {n_nodule}'
code_output = 'code outputs/c16 avg decision fusion img rad txt'

all_input_files = os.listdir(score_folder)
input_files = [filename for filename in all_input_files if filename.endswith('.pkl') and 'score' in filename]

IMG_RAD_TEXT_all_proba = []
IMG_RAD_TEXT_all_y_true = []

all_scores = []
for idx, filename in enumerate(input_files):
    print(filename)
    with open(f'{score_folder}/{filename}', 'rb') as file:
        a_score = pickle.load(file)
        IMG_RAD_TEXT_all_proba.append(a_score['y_proba_all'])
        IMG_RAD_TEXT_all_y_true.append(a_score['y_true_all'])
        
        
IMG_proba, RAD_proba, TXT_proba = IMG_RAD_TEXT_all_proba
IMG_y_true, RAD_y_true, TXT_y_true = IMG_RAD_TEXT_all_y_true


'''_________Checking the Sequence of y_tyre for all feature IS EXACTLY SAME__________________'''
if IMG_y_true == RAD_y_true == TXT_y_true: 
    print('all y true are same')

y_true = IMG_y_true


prob_scores = np.array([IMG_proba, RAD_proba, TXT_proba]) # , IMG_proba, RAD_proba, TXT_proba

# Calculate F1-score using equal weights (1/3, 1/3, 1/3)
equal_weights = np.array([1/3, 1/3, 1/3]) # , 1/3
equal_weighted_avg_prob = np.average(prob_scores, axis=0, weights=equal_weights)
equal_final_prediction = (equal_weighted_avg_prob > 0.5).astype(int)
equal_f1_score = f1_score(y_true, equal_final_prediction)

np.save(os.path.join(code_output, f"{modality_type}_avg_proba_{n_nodule}.npy"), equal_weighted_avg_prob) # update c15
np.save(os.path.join(code_output, f"{modality_type}_avg_y_true_{n_nodule}.npy"), equal_final_prediction) # update c15


print("F1 Score with Equal Weights (1/2, 1/2):", equal_f1_score)



"""============================ Fold Wise Prediction ======================="""
metrics = {'Accuracy': [], 'Precision': [], 'Recall (Sensitivity)': [], 'Specificity': [], 'F1 Score': [], 'AUROC': []}

y_true = y_true
y_pred = equal_final_prediction.tolist() # update c15
y_proba = [round(float(x), 5) for x in equal_weighted_avg_prob] # update c15

for i in range(5):
    
    fold_size = equal_weighted_avg_prob.shape[0]//5
    
    y_true_fold = y_true[i*0 : (i+1)*fold_size]
    y_pred_fold = y_pred[i*0 : (i+1)*fold_size]
    y_proba_fold = y_proba[i*0 : (i+1)*fold_size]
    
    cm = confusion_matrix(y_true_fold, y_pred_fold)
    TN, FP, FN, TP = cm.ravel()

    metrics['Accuracy'].append(accuracy_score(y_true_fold, y_pred_fold))
    metrics['Precision'].append(precision_score(y_true_fold, y_pred_fold))
    metrics['Recall (Sensitivity)'].append(recall_score(y_true_fold, y_pred_fold))
    metrics['Specificity'].append(TN / (TN + FP))
    metrics['F1 Score'].append(f1_score(y_true_fold, y_pred_fold))
    metrics['AUROC'].append(roc_auc_score(y_true_fold, y_proba_fold))

# Create results DataFrame
results_df = pd.DataFrame({
    metric: [f"{np.mean(scores)*100:.2f} ± {np.std(scores)*100:.2f}"]
    for metric, scores in metrics.items()
})

# Save to CSV
results_df.to_csv(f"{code_output}/{modality_type}_avg_prediction_{n_nodule}.csv", index=False, encoding='utf-8-sig')

# Display
print(results_df)
