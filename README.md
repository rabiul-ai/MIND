# MIND
A novel Multimodal Interpretable Nodule Diagnosis framework for benign malignant lung nodule classification

# Highlights:
* First **multimodal** model that integrates CT scan images, radiological attributes, and textual nodule descriptions to align clinical diagnosis  
* Clinically informative **interpretability** using SHAP values for radiological attributes and word-level importance for nodule classification  
* Text-based classification using a domain-specific pretrained **PubMedBERT** model on **anatomy-aware** generated textual descriptions
* Comprehensive ablation studies, including three-way **cross-attention** and **contrastive loss**–based **joint embedding** in a shared latent space  

# Multimodal framework:
<img width="3192" height="2501" alt="workflow" src="https://github.com/user-attachments/assets/8a983a03-0baa-4337-a936-c551ea783d30" />


# Multimodal Fusion:
<img width="3192" height="946" alt="fusion" src="https://github.com/user-attachments/assets/1801454d-e197-4265-892a-885aaf1d6010" />

# Result:
<img width="4800" height="2400" alt="bar plot comp IRT set A" src="https://github.com/user-attachments/assets/901cf3e4-adc8-4bed-a4bd-5839da8df954" />

# Dataset: 
LIDC-IDRI  
Text description is generated
