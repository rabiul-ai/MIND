# Date: July, 2025 
# Author: Md Rabiul Islam, PhD Student, ECEN, TAMU
# Task: Preprocess the LIDC-IDRI dataset. Includes:
#           1. Save the nodule images (2D views and 3D) # Not maks like as previous
#           2. Save CT metadata containing CT text description
#           3. Save nodule metadata containing nodule attributes and text description
#           4. Update: The preprocessing, Process 1-6 are ordered properly
# Update than c1: in c1 saving only eligible 1758 nodules. But here saving all 2625 nodules

"""========================== Part A: Import Libraries ====================="""
import pylidc as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from statistics import median_high
from scipy.ndimage import zoom
from text_utility import generate_nodule_description, get_radiomic_features, generate_centroid_location_description, get_lung_region, generate_a_nodule_description_completely    


""" ==========================Part B: Directories =========================="""
input_data_path = r"C:\Rabiul\1. PhD Research\7. Summer 2024\Coding\Dataset\LIDC-IDRI"
# input_data_path = r"C:\Rabiul\1. PhD Research\7. Summer 2024\Coding\Dataset\Only 10 LIDC-IDRI" # for testing code

code_outputs = 'code outputs/c1 data preprocessing 2625 nodules'

output_2d_images = f"{code_outputs}/Data/2D Nodule Views"
output_3d_images = f"{code_outputs}/Data/3D Nodule Volumes"
output_2d_9slices_img = f"{code_outputs}/Data/2D Nodule Views Arrary"

# Create directories if they do not exist
for directory in [output_2d_images, output_3d_images, output_2d_9slices_img]:
    os.makedirs(directory, exist_ok=True)


"""======================= Part C: Set Parameters =========================="""
NEW_SPACING = (1, 1, 1)
NODULE_SHAPE = (32, 32, 32)
HU_LEVEL = -200
HU_WINDOW = 1200


"""==================== Part D: User Defined Functions ====================="""
def crop_nodule(img_volume, centroid, croping_shape):
    x_center, y_center, z_center = map(int, np.round(centroid))
    
    # Define the crop boundaries (centered around the new centroid)
    crop_x, crop_y, crop_z = croping_shape
    half_x = int(crop_x / 2)
    half_y = int(crop_y / 2)
    half_z = int(crop_z / 2)
    
    # Define the boundaries of the cube to extract (ensure they are within volume bounds)
    x_min = max(x_center - half_x, 0)
    x_max = min(x_center + half_x, img_volume.shape[0]) # + 1 because desired shape is 71, a odd number
    y_min = max(y_center - half_y, 0)
    y_max = min(y_center + half_y, img_volume.shape[1])
    z_min = max(z_center - half_z, 0)
    z_max = min(z_center + half_z, img_volume.shape[2])
    
    cropped_nodule = img_volume[x_min:x_max, y_min:y_max, z_min:z_max]
    return cropped_nodule


def calculate_malignancy(nodule):
    # Calculate the malignancy of a nodule with the annotations made by 4 doctors. Return median high of the annotated cancer, True or False label for cancer
    # if median high is above 3, we return a label True for cancer
    # if it is below 3, we return a label False for non-cancer
    # if it is 3, we return ambiguous
    list_of_malignancy =[]
    for annotation in nodule:
        list_of_malignancy.append(annotation.malignancy)
    malignancy = median_high(list_of_malignancy)
    if  malignancy > 3:
        return malignancy, 'Y'
    elif malignancy < 3:
        return malignancy, 'N'
    else:
        return malignancy, 'U'


def get_nine_slices(cube):
    # Ensure cube shape is (32, 32, 32) by padding with zeros if necessary
    target_shape = (32, 32, 32)
    current_shape = cube.shape
    if current_shape != target_shape:
        pad_width = [(0, max(0, t - s)) for s, t in zip(current_shape, target_shape)]
        cube = np.pad(cube, pad_width, mode='constant', constant_values=0)
        cube = cube[:32, :32, :32]  # In case cube was larger
    
    cube_size = cube.shape[0]
    mid_x = mid_y = mid_z = cube_size // 2
    
    # Middle 3 slices 
    coronal = cube[:, mid_y, :]     # front-view
    sagittal = cube[mid_x, :, :]    # left-view
    axial = cube[:, :, mid_z]       # top-view
    
    # Diagonal 6 slices 
    plane_1 = np.array([cube[i, i, :] for i in range(cube_size)])
    plane_2 = np.array([cube[i, cube_size-1-i, :] for i in range(cube_size)])

    plane_3 = np.transpose(np.array([cube[i, :, i] for i in range(cube_size)]))
    plane_4 = np.transpose(np.array([cube[i, :, cube_size-1-i] for i in range(cube_size)]))

    plane_5 = np.array([cube[:, i, i] for i in range(cube_size)])
    plane_6 = np.array([cube[:, i, cube_size-1-i] for i in range(cube_size)])
    
    nine_slices = np.stack((coronal, sagittal, axial, plane_1, plane_2, plane_3, plane_4, plane_5, plane_6), axis=0)
    return nine_slices


def normalize_volume_0_1(volume):
    max_value, min_value = np.max(volume), np.min(volume)
    normalized_volume = (volume - min_value) / (max_value - min_value)
    normalized_volume = np.clip(normalized_volume, 0, 1)
    return normalized_volume


def determine_overall_CT_malignancy(malignancies):
    if any(score in [4, 5] for score in malignancies):
        return 'Y'  # Malignant
    elif all(score in [1, 2] for score in malignancies):
        return 'N'  # Benign
    else:
        return 'U'  # Uncertain


def append_a_nodule_info():
    nodule_info = (
        nodule_count_all, nodule_selected, why_not_selected, nodule_id, pid, avg_nodule_diameter, mask_shape_mm,
        ct_shape, original_centroid_a_nodule, anatomic_region, subtlety, internalStructure, calcification, 
        sphericity, margin, lobulation, spiculation, texture,
        nodule_text_description, malignancy, cancer_label
        )
    all_nodule_info.append(nodule_info)


"""======================== Part E:  Main Code Starts ======================"""
patient_ids = os.listdir(input_data_path) 

all_ct_info = []
all_nodule_info = []

pid_not_dia_3mm = []
pid_not_32_cube = []
pid_not_vol_180_xyz_4 = []

# count all noduels, regardless the nodule is selected in our experiment or not. all 2625 nodule counting.
nodule_count_all = 0
# nodule_count_selected = 0

# iterating all the patients_________________________________________
for pid in tqdm(patient_ids):
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first() # pid
    nodules_annotation = scan.cluster_annotations()
    
    """======================== Part F: CT Processing ======================"""
    '''____________________ Process 1: Load DICOM data _____________________'''
    dicom_slices = scan.load_all_dicom_images() 
    img_dicom = np.stack([s.pixel_array for s in dicom_slices]) # z,x,y
    img_dicom = np.transpose(img_dicom, (1, 2, 0)) # from z,x,y to x,y,z
    
    '''__________________ Process 2: Convert DICOM to HU  __________________'''
    slope = int(dicom_slices[10].RescaleSlope)
    intercept = int(dicom_slices[10].RescaleIntercept)
    img_HU = img_dicom * slope + intercept
    
    '''_____________ Process 3: Resampling to Isotropic Spacing  ___________'''
    original_spacing = (scan.pixel_spacing, scan.pixel_spacing, scan.slice_spacing)
    resample_factors = np.array(original_spacing) / np.array(NEW_SPACING)
    img_HU_resampled = zoom(img_HU, resample_factors, order=3)  # 'order=3' for spline interpolation

    '''__________________ Process 4: Apply Lung HU Window  _________________'''
    window_min = HU_LEVEL - HU_WINDOW // 2  # integer division //
    window_max = HU_LEVEL + HU_WINDOW // 2
    img_HU_resampled_wind = np.clip(img_HU_resampled, window_min, window_max)
    
    '''__________________ Process 5: Normalize to [0, 1] ___________________'''
    img_normalized = normalize_volume_0_1(img_HU_resampled_wind)  
    
    # whole CT information __________________________   
    ct_shape = img_dicom.shape # x,y,x
    ct_spacing = (scan.pixel_spacing, scan.pixel_spacing, scan.slice_spacing)
    num_nodules = len(nodules_annotation)

    # # NODULE Information____________
    diameters = []
    malignancies = []
    original_centroids = []
    all_nodule_description = []
    
    # if at least 1 nodule exist ______________________
    if len(nodules_annotation) > 0:  

        """================= Part G: Nodule Processing ====================="""
        for nodule_idx, a_nodule in enumerate(nodules_annotation): # Iterating Through Each Nodule
            nodule_count_all += 1

            # nodule diameter, centroid and mask information_____
            nodule_dias = []  
            original_centroid_4radiologists = np.empty((0, 3))
            mask_shapes = []
            
            # iterating through 4 RADIOLOGISTS annotations
            for radiologist_idx, annotation in enumerate(a_nodule): 
                # diameter______
                nodule_size = annotation.diameter                
                nodule_dias.append(round(nodule_size, 2))     

                # centroid_______
                original_centroid_1radiologist = a_nodule[radiologist_idx].centroid
                original_centroid_4radiologists = np.vstack((original_centroid_4radiologists, original_centroid_1radiologist))
                
                # mask shape ______
                mask = a_nodule[radiologist_idx].boolean_mask()
                mask_shapes.append(mask.shape)
                
                
            # diameter averaging ______
            avg_nodule_diameter = round(sum(nodule_dias)/len(nodule_dias), 2) # make avergae diameter of
            diameters.append(avg_nodule_diameter)
            
            # centroid averaging _______
            original_centroid_a_nodule = np.mean(original_centroid_4radiologists, axis=0) # averaging 
            original_centroid_a_nodule = np.round(original_centroid_a_nodule).astype(int) # rounding as int
            original_centroid_a_nodule = tuple(original_centroid_a_nodule)
            original_centroids.append(original_centroid_a_nodule) # appending in main list
            
            # mask shape averaging______
            mask_shape = np.mean(mask_shapes, axis=0)
            mask_shape_mm = np.array(mask_shape) * resample_factors
            mask_shape_mm = tuple(np.round(mask_shape_mm, 2))

            # extract radiomic features
            radiomic_features = get_radiomic_features(a_nodule)
            subtlety, internalStructure, calcification, sphericity, margin, lobulation, spiculation, texture = radiomic_features
            
            # malignancy_____
            malignancy, cancer_label = calculate_malignancy(a_nodule)
            malignancies.append(malignancy)
            
            # generating CT text description 
            a_nodule_text_description = generate_nodule_description(radiomic_features, nodule_idx)
            all_nodule_description.append(a_nodule_text_description)

            # generating a single nodule text description, COMPLETE nodule description
            anatomic_region = get_lung_region(ct_shape, original_centroid_a_nodule)
            nodule_text_description = generate_a_nodule_description_completely(avg_nodule_diameter, anatomic_region, radiomic_features)
            
            
            '''__________________ Process 6: Cropping nodule _______________________'''
            # Adjust the centroid to the new resampled volume
            resampled_centroid = np.array(original_centroid_a_nodule) * resample_factors            
            nodule_vol = crop_nodule(img_volume = img_normalized, centroid = resampled_centroid, croping_shape = NODULE_SHAPE)
            
            # By default setting__________
            nodule_selected = 'Yes'
            why_not_selected = ''
            
            """============ Part H: Nodule Selection Criterias ============="""
            # Nodule selection criteria 1: reported nodule diameter have to be minimum 3 mm_______________________________________
            if avg_nodule_diameter < 3:
                pid_not_dia_3mm.append(pid)
                nodule_selected = 'No'
                why_not_selected = 'diameter < 3 mm'
            

            # Nodule selection criteria 2: cropped nodule should have shape 32, 32, 32
            if nodule_vol.shape != (32, 32, 32): # update 3: for patient 463 32, 30, 32 __________________
                pid_not_32_cube.append(pid)
                nodule_selected = 'No'
                why_not_selected = 'cube shape not 32'
            
            # Nodule selection criteria 3: nodule mask shape have to be big enough (for example volume > 180 mm^3 and in any x, y or z > 4 mm)
            nod_mask_x, nod_mask_y, nod_mask_z = mask_shape_mm
            nod_volume = nod_mask_x * nod_mask_y * nod_mask_z
            if nod_volume < 180 or min(nod_mask_x, nod_mask_y, nod_mask_z) < 4:
                pid_not_vol_180_xyz_4.append(pid)
                nodule_selected = 'No'
                why_not_selected = 'vol < 180 or any_xyz < 4'
            
            
            """============ Part I: Save Nodule Imgs and Arrays ============"""
            # If all above 3 criteris satisfied or not, save nodule.
            
            # saving 9 Slices as Image ______________________________
            nodule_nine_slices = get_nine_slices(nodule_vol)
            for i in range(9):
                nodule_dia_int = int(round(avg_nodule_diameter))
                nodule_name = "nid_{}_sel_{}_s_{}_dia_{}_p_{}_n_{}_m_{}_c_{}.png".format(nodule_count_all, nodule_selected, i+1, nodule_dia_int, pid[-4:], nodule_idx+1, malignancy, cancer_label)
                img = nodule_nine_slices[i, :, :]
                nodule_path = os.path.join(output_2d_images, nodule_name)
                plt.imsave(nodule_path, img, cmap='gray')
            
            # saving 3D array_______________________________________
            nodule_name = "n3D_{}_sel_{}_dia_{}_p_{}_n_{}_m_{}_c_{}.npy".format(nodule_count_all, nodule_selected, nodule_dia_int, pid[-4:], nodule_idx+1, malignancy, cancer_label)
            nodule_path = os.path.join(output_3d_images, nodule_name)
            np.save(nodule_path, nodule_vol)
            
            # Saving 9 Slices as Array ______________________________
            nodule_name = "n2D_{}_sel_{}_dia_{}_p_{}_n_{}_m_{}_c_{}.npy".format(nodule_count_all, nodule_selected, nodule_dia_int, pid[-4:], nodule_idx+1, malignancy, cancer_label)
            nodule_path = os.path.join(output_2d_9slices_img, nodule_name)
            np.save(nodule_path, nodule_nine_slices)

            # Saving nodule metadata__________________________________________________
            nodule_id = nodule_count_all
            append_a_nodule_info()
            print(f"\nSaved Nodule nid = {nodule_count_all}")
            

    """================== Part J: Generating Full CT Info =================="""
    # Whole CT information making and saving___________________________________________________________________            
    ct_cancer_label = determine_overall_CT_malignancy(malignancies)

    # Combine all nodules text descriptions
    location_description = generate_centroid_location_description(ct_shape, original_centroids) # Add centroid location information to the description
    full_ct_description = f"Lung CT scan contains {num_nodules} nodules with diameters {', '.join(f'{d} mm' for d in diameters)}; {location_description} " + " ".join(all_nodule_description)
    
    # just for 1 nodule text would be singular. like nodule, not nodules.
    if num_nodules == 1:
        full_ct_description = f"Lung CT scan contains {num_nodules} nodule with diameter {', '.join(f'{d} mm' for d in diameters)}; {location_description} " + " ".join(all_nodule_description)

    # if the CT doesn't have any nodule
    if num_nodules == 0:
        any_nodule = 'N'
        full_ct_description = "No nodules are present in the lung CT scan."
    else:
        any_nodule = 'Y'

    # full information about a whole CT
    ct_info = (pid, any_nodule, num_nodules, ct_cancer_label, ct_shape, ct_spacing, full_ct_description)
    all_ct_info.append(ct_info)

    print(f"\nSaved CT information pid = {pid}")


"""======================== Part K: Saving Metadata ======================="""
# Saving CT metadata
colum_name_ct_info = [
    'patient ID', 'nodule present', 'CT total nodules', 'CT cancer label', 
    'CT 3D shape', 'CT spacing', 'CT text description'
    ]
ct_info_df = pd.DataFrame(all_ct_info, columns=colum_name_ct_info)
ct_info_df.to_excel(f'{code_outputs}/LIDC-IDRI CT metadata.xlsx', index=False) 

# Saving nodule metadata
colum_name_nodule_info = ['all nodule count', 'nodule selected', 'why not selected', 'nodule ID', 'patient ID', 'nodule diameter (mm)', 'nodule mask shape (mm)',
                          'CT shape', 'nodule centroid', 'anatomic position', 'subtlety', 'internal structure', 'calcification',
                          'sphericity', 'margin', 'lobulation', 'spiculation', 'textures', 
                          'nodule description', 'malignancy', 'cancer label']
nodule_info_df = pd.DataFrame(all_nodule_info, columns=colum_name_nodule_info)
nodule_info_df.to_excel(f'{code_outputs}/LIDC-IDRI nodule metadata.xlsx', index=False) 
