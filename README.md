# MIND
A novel Multimodal Interpretable Nodule Diagnosis framework for benign malignant lung nodule classification

# Highlights:
* First **multimodal** model that integrates CT scan images, radiological attributes, and textual nodule descriptions  
* Clinically informative **interpretability** using SHAP values for radiological attributes and word-level importance for nodule classification  
* Text-based classification using a domain-specific pretrained **PubMedBERT** model on **anatomy-aware** generated textual descriptions
* Comprehensive ablation studies, including three-way **cross-attention** and **contrastive loss**–based **joint embedding** in a shared latent space  

# Multimodal framework:
<img width="3178" height="2590" alt="Multimodal Framework only" src="https://github.com/user-attachments/assets/53828ffb-f9f2-4cce-9c2a-00da6d0179d1" />

# Multimodal Fusion:
<img width="3101" height="973" alt="Fusion techniques" src="https://github.com/user-attachments/assets/fdaebfed-3abf-40c7-91d1-e9335a4a033b" />

# Result:
<img width="4800" height="2400" alt="bar plot comp IRT set A" src="https://github.com/user-attachments/assets/901cf3e4-adc8-4bed-a4bd-5839da8df954" />

# Dataset: 
LIDC-IDRI  
Text description is generated
