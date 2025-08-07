'''
Date: July 11, 2025
Task: Image, Radiological Decision Fusion
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
n_nodule = 1385
modality_type = 'IMG_RAD'


"""==============Step 1: Searching Weights for Fusion of Image, Radiological and Text ============"""
"""______________________ Loading Scores ____________________________"""
# input file directories
score_folder = f'code inputs/c13 img rad text predictions {n_nodule}'
code_output = 'code outputs/c15 avg decision fusion'

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

# # as type of GLCM and HOG y_ture was int64. convert it to int32
# GLCM_y_true = list(map(np.int32, GLCM_y_true))
# HOG_y_true = list(map(np.int32, HOG_y_true))

'''_________Checking the Sequence of y_tyre for all feature IS EXACTLY SAME__________________'''
if IMG_y_true == RAD_y_true == TXT_y_true: 
    print('all y true are same')

y_true = IMG_y_true


prob_scores = np.array([IMG_proba, RAD_proba]) # , IMG_proba, RAD_proba, TXT_proba


# """__________________________ Bayesian Optimization ______________________"""
# # Stack the probability scores for IMG_proba, RAD_proba, TXT_proba


# # Define the optimization function to maximize the F1-score
# def optimize_weights(weights):
#     weights = np.array(weights)
#     weights /= weights.sum()  # Ensure weights sum to 1
#     weighted_avg_prob = np.average(prob_scores, axis=0, weights=weights)
#     y_pred = (weighted_avg_prob > 0.5).astype(int)
#     f1 = f1_score(y_true, y_pred)
#     return -f1  # Minimize negative F1-score

# # Define the search space for weights
# search_space = [Real(0, 1), Real(0, 1)] # , Real(0, 1)

# # Perform Bayesian optimization
# result = gp_minimize(optimize_weights, search_space, n_calls=100, random_state=0)

# # Get the optimal weights
# optimal_weights = result.x
# optimal_weights = np.array(optimal_weights)
# optimal_weights /= optimal_weights.sum()  # Ensure weights sum to 1

# # Calculate the weighted average probabilities with optimal weights
# weighted_avg_prob = np.average(prob_scores, axis=0, weights=optimal_weights)
# final_prediction = (weighted_avg_prob > 0.5).astype(int)

# print("Optimal Weights:", optimal_weights)
# print("Final F1 Score:", f1_score(y_true, final_prediction))



# Calculate F1-score using equal weights (1/3, 1/3, 1/3)
equal_weights = np.array([1/2, 1/2]) # , 1/3
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
