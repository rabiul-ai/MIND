import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, EfficientNetV2B0, InceptionV3, MobileNet, NASNetMobile, DenseNet121, ConvNeXtTiny
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Lambda
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc



def create_model(model_name, n_freeze):
    # Load Pretrained Model________________________________________________
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False)
    elif model_name == 'VGG19':
        base_model = VGG19(weights='imagenet', include_top=False)
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False)
    elif model_name == 'EfficientNetV2B0':
        base_model = EfficientNetV2B0(weights='imagenet', include_top=False)
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False)
    elif model_name == 'MobileNet':
        base_model = MobileNet(weights='imagenet', include_top=False)
    elif model_name == 'NASNetMobile':
        base_model = NASNetMobile(weights='imagenet', include_top=False)
    elif model_name == 'DenseNet121':
        base_model = DenseNet121(weights='imagenet', include_top=False)
    elif model_name == 'ConvNeXtTiny':
        base_model = ConvNeXtTiny(weights='imagenet', include_top=False)
    else:
        raise ValueError("Model not supported")
        
    
    '''Very Important: Images Need to be resized as input shape of the model
    For example in our case from 32*32 to 224*224. Important for good accuracy'''
    img_resize = (224, 224) #base_model.input_shape[1:3]    # to_res = (224, 224)
    # print(model_name, base_model)
    # print(img_resize)

    # Decide how many layes want to freeze__________________________________   
      
    # for layer in base_model.layers[:-4]:  # Unfreeze last 4 layers________
    # layer.trainable = False
    
    for layer in base_model.layers[:-n_freeze]:  # for example, n_freeze = 10
        layer.trainable = False

    # Making Complete Model_______________________________________________
    model = Sequential()
    model.add(Lambda(lambda image: tf.image.resize(image, img_resize)))
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def adjust_probability_list(diameters, model_probs, kde_benign, kde_malignant, alpha=1.0, prior_benign=0.5, prior_malignant=0.5):
    """
    Adjust a list of model probabilities based on their corresponding nodule diameters, controlling the dependence using alpha.
    
    Parameters:
        diameters (list): List of diameters of nodules.
        model_probs (list): List of model probabilities for each nodule.
        kde_benign: KDE function for benign nodule size distribution.
        kde_malignant: KDE function for malignant nodule size distribution.
        alpha (float): The factor controlling the dependence on diameter (0 = no effect, 1 = full effect).
        prior_benign (float): Prior probability for benign nodules.
        prior_malignant (float): Prior probability for malignant nodules.
    
    Returns:
        list: List of adjusted probabilities.
    """
    adjusted_probs = []

    for diameter, model_prob in zip(diameters, model_probs):
        # KDE values for benign and malignant at the given diameter
        p_d_given_benign = kde_benign(diameter)
        p_d_given_malignant = kde_malignant(diameter)
        
        # Total probability P(d) (normalizing factor)
        p_d = p_d_given_benign * prior_benign + p_d_given_malignant * prior_malignant
        
        # Posterior probabilities using Bayes' theorem
        posterior_benign = (p_d_given_benign * prior_benign) / p_d
        posterior_malignant = (p_d_given_malignant * prior_malignant) / p_d

        # Adjust the model's probability based on the priors and alpha
        adjusted_prob = (posterior_malignant * model_prob) / (posterior_benign * (1 - model_prob) + posterior_malignant * model_prob)
        
        # Blend the original model probability with the adjusted one using alpha
        final_prob = alpha * adjusted_prob + (1 - alpha) * model_prob
        final_prob = round(final_prob[0], 2)
        adjusted_probs.append(final_prob)
    
    return adjusted_probs

def evaluate_and_print_performance(fold, y_test, y_pred, y_pred_prob, accuracies, precisions, recalls, f1_scores, roc_aucs,  sensitivities, specificities, scores):
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1]) # [:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = recall  # Sensitivity is the same as recall
    specificity = tn / (tn + fp)

              
    # Append metrics to lists
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    roc_aucs.append(roc_auc)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
    # Print metrics and confusion matrix
    # print('After Diameter Adjustment___________________')
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    
    
    '''___________________Making a Dictionary for Saving________________'''
    scores[f'fold = {fold+1}'] = {}
    scores[f'fold = {fold+1}']['Accuracy'] = f"{accuracy:.2%}"
    scores[f'fold = {fold+1}']['F1 Score'] = f"{f1:.2%}"
    scores[f'fold = {fold+1}']['AUC     '] = f"{roc_auc:.2%}"
    scores[f'fold = {fold+1}']['Confusion Matrix'] = np.array([[tn, fp], [fn, tp]])
    scores[f'fold = {fold+1}']['Sensitivity'] = f"{sensitivity:.2%}"
    scores[f'fold = {fold+1}']['Specificity'] = f"{specificity:.2%}"
    scores[f'fold = {fold+1}']['Precision  '] = f"{precision:.2%}"
    scores[f'fold = {fold+1}']['Recall     '] = f"{recall:.2%}"

    return accuracies, precisions, recalls, f1_scores, roc_aucs,  sensitivities, specificities, scores




def get_mean_performance_of_all_folds(performance_metrices, model_name = 'Original', results = []):
    
    accuracies, precisions, recalls, f1_scores, roc_aucs, sensitivities, specificities, scores = performance_metrices
    # Calculate mean and standard deviation of metrics for the current model
    mean_accuracy = np.mean(accuracies)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1_score = np.mean(f1_scores)
    mean_roc_auc = np.mean(roc_aucs)
    mean_sensitivity = np.mean(sensitivities)
    mean_specificity = np.mean(specificities)

    std_accuracy = np.std(accuracies)
    std_precision = np.std(precisions)
    std_recall = np.std(recalls)
    std_f1_score = np.std(f1_scores)
    std_roc_auc = np.std(roc_aucs)
    std_sensitivity = np.std(sensitivities)
    std_specificity = np.std(specificities)

    results.append({
        'Model': model_name,
        'Accuracy': f"{mean_accuracy*100:.2f} ± {std_accuracy*100:.2f}",
        'Precision': f"{mean_precision*100:.2f} ± {std_precision*100:.2f}",
        'Recall (Sensitivity)': f"{mean_recall*100:.2f} ± {std_recall*100:.2f}",
        'Specificity': f"{mean_specificity*100:.2f} ± {std_specificity*100:.2f}",
        'F1 Score': f"{mean_f1_score*100:.2f} ± {std_f1_score*100:.2f}",
        'AUROC': f"{mean_roc_auc*100:.2f} ± {std_roc_auc*100:.2f}"
        
    })


    '''____________________ Making Dictionary Avg. Score ____________________'''
    scores['All Folds Avg.'] = {}    

    scores['All Folds Avg.']['Accuracy   '] = f"{mean_accuracy*100:.2f} ± {std_accuracy*100:.2f}"
    scores['All Folds Avg.']['Precision  '] = f"{mean_precision*100:.2f} ± {std_precision*100:.2f}"
    scores['All Folds Avg.']['Recall (Sensitivity)'] = f"{mean_recall*100:.2f} ± {std_recall*100:.2f}"
    scores['All Folds Avg.']['Specificity'] = f"{mean_specificity*100:.2f} ± {std_specificity*100:.2f}"
    scores['All Folds Avg.']['F1 Score'] = f"{mean_f1_score*100:.2f} ± {std_f1_score*100:.2f}"
    scores['All Folds Avg.']['AUROC   '] = f"{mean_roc_auc*100:.2f} ± {std_roc_auc*100:.2f}"
    return results, scores



def save_scores_in_txt_file(output_path, code_no, scores, alpha, model_name, dia = 'No'):
    file_path = f"{output_path}/{code_no} Model_{model_name}_diameter_{dia}_models_scores.txt"
    with open(file_path, 'w') as file:
        file.write(f"================== PERFORMANCE METRICS FOR MODEL = {model_name} Diameter = {dia}: ================== \n\n")
        
        for fold_name, metrics in scores.items():
            file.write(f"Model: Alpha_{alpha}, Fold: {fold_name} _______________\n")
            
            for metric, value in metrics.items():
                if metric == 'Confusion Matrix':
                    file.write(f"{metric}:\n{value}\n")
                else:
                    file.write(f"{metric}: {value}\n")
            file.write("\n")
        
        file.write("\n")


if __name__ == "__main__":
    pass