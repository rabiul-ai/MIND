# This file contains the utility functions for generating text descriptions of nodules.

import numpy as np

def get_lung_region(ct_shape, nodule_centroid):

    x, y, z = nodule_centroid
    # Normalize coordinates
    x_norm = x / ct_shape[0]  
    y_norm = y / ct_shape[1]
    z_norm = z / ct_shape[2]
    # print(original_3d_shape)
    
    # Determine superior/inferior position
    if z_norm < 0.33:
        vertical = "upper"
    elif z_norm < 0.66:
        vertical = "middle"
    else:
        vertical = "lower"
        
    # Determine anterior/posterior position
    if y_norm < 0.5:
        ap = "anterior"
    else:
        ap = "posterior"
        
    # Determine left/right position
    if x_norm < 0.5:
        lr = "left"
    else:
        lr = "right"
        
    # Determine specific lobe based on coordinates
    if vertical == "upper":
        if lr == "right":
            if x_norm < 0.3:
                lobe = "right upper lobe"
            else:
                lobe = "right middle lobe"
        else:
            lobe = "left upper lobe"
    elif vertical == "middle":
        if lr == "right":
            lobe = "right middle lobe"
        else:
            lobe = "left lower lobe"
    else:  # lower
        if lr == "right":
            lobe = "right lower lobe"
        else:
            lobe = "left lower lobe"
            
    return f"{ap} {lobe}"


def generate_centroid_location_description(ct_shape, centroids):
      
    """
    Generate text description of nodule locations based on their centroids using clinical lung regions
    
    Args:
        centroids: List of nodule centroid coordinates
        
    Returns:
        str: Description of nodule locations
    """
    if not centroids:
        return ""
        
    
        
    # Group nodules by region
    regions = {}
    for i, a_nodule_centroid in enumerate(centroids):
        region = get_lung_region(ct_shape, a_nodule_centroid) # x, y, z
        if region not in regions:
            regions[region] = []
        regions[region].append(i+1)  # +1 for 1-based nodule numbering
        
    # Generate description
    location_desc = []
    for region, nodules in regions.items():
        if len(nodules) > 1:
            location_desc.append(f"nodules {', '.join(map(str, nodules))} in the {region}")
        else:
            location_desc.append(f"nodule {nodules[0]} in the {region}")
            
    return " with " + " and ".join(location_desc) + "."


def get_radiomic_features(nodule):
    """
    Returns radiomic features.
    
    Args:
        nodule: A list of nodule annotations from LIDC-IDRI (one from each radiologist)
        
    Returns:
        int: radiomic features 
    """
    # Get the consensus attributes by taking the median value from all radiologists
    subtlety = int(np.median([ann.subtlety for ann in nodule]))
    internalStructure = int(np.median([ann.internalStructure for ann in nodule]))
    calcification = int(np.median([ann.calcification for ann in nodule]))
    sphericity = int(np.median([ann.sphericity for ann in nodule]))
    margin = int(np.median([ann.margin for ann in nodule]))
    lobulation = int(np.median([ann.lobulation for ann in nodule]))
    spiculation = int(np.median([ann.spiculation for ann in nodule]))
    texture = int(np.median([ann.texture for ann in nodule]))

    radiomic_features = subtlety, internalStructure, calcification, sphericity, margin, lobulation, spiculation, texture

    return radiomic_features


def generate_nodule_description(radiomic_features, nodule_idx):
    """
    Generate a text description for a nodule based on its attributes given in radiomic features.
    
    Args:
        nodule: attributes given in radiomic features
        
    Returns:
        str: A descriptive text about the nodule
    """
    # Get the consensus attributes by taking the median value from all radiologists
    subtlety, internalStructure, calcification, sphericity, margin, lobulation, spiculation, texture = radiomic_features
    
    # Create the description
    description = f"Nodule {nodule_idx + 1} is "  # nodule_idx = 0 means first nodule
    
    # Add subtlety
    subtlety_desc = {
        1: "extremely subtle",
        2: "moderately subtle",
        3: "fairly subtle",
        4: "moderately obvious",
        5: "obvious"
    }.get(subtlety, "unknown")
    description += f"{subtlety_desc}, "
    
    # Add internal structure
    internal_structure_desc = {
        1: "soft tissue composition",
        2: "fluid composition",
        3: "fat composition",
        4: "air composition"
    }.get(internalStructure, "unknown internal structure")
    description += f"with {internal_structure_desc}, "
    
    # Add calcification
    calc_desc = {
        1: "popcorn calcification",
        2: "laminated calcification",
        3: "solid calcification",
        4: "non-central calcification",
        5: "central calcification",
        6: "no calcification"
    }.get(calcification, "unknown")
    description += f"showing {calc_desc}, "
    
    # Add shape characteristics
    shape_desc = []
    if sphericity:
        sphericity_desc = {
            1: "linear",
            2: "ovoid or linear",
            3: "ovoid",
            4: "ovoid or round",
            5: "round"
        }.get(sphericity, "unknown")
        shape_desc.append(f"{sphericity_desc} shape")
    
    if margin:
        margin_desc = {
            1: "poorly defined",
            2: "near poorly defined",
            3: "medium",
            4: "near sharp",
            5: "sharp"
        }.get(margin, "unknown")
        shape_desc.append(f"{margin_desc} margins")
    
    if lobulation:
        lobulation_desc = {
            1: "no lobulation",
            2: "nearly no lobulation",
            3: "moderate lobulation",
            4: "near marked lobulation",
            5: "marked lobulation"
        }.get(lobulation, "unknown")
        shape_desc.append(f"{lobulation_desc}")
    
    if spiculation:
        spiculation_desc = {
            1: "no spiculation",
            2: "nearly no spiculation",
            3: "moderate spiculation",
            4: "near marked spiculation",
            5: "marked spiculation"
        }.get(spiculation, "unknown")
        shape_desc.append(f"{spiculation_desc}")
    
    if shape_desc:
        description += "with " + ", ".join(shape_desc) + ", "
    
    # Add texture
    if texture:
        texture_desc = {
            1: "non-solid or Ground-Glass Opacity",
            2: "non-solid or mixed",
            3: "part solid or mixed",
            4: "solid or mixed",
            5: "solid"
        }.get(texture, "unknown")
        description += f"and {texture_desc} texture."
    
    return description 


def generate_a_nodule_description_completely(diameter, anatomic_region, radiomic_features):
    """
    Generate a text description for a nodule based on its size, anatomic region, and radiomic features.
    
    Args:
        diameter (float): Diameter of the nodule in mm
        anatomic_region (str): Anatomical region description
        radiomic_features (tuple): Radiomic features (subtlety, internalStructure, calcification, sphericity, margin, lobulation, spiculation, texture)
    
    Returns:
        str: A descriptive text about the nodule
    """
    subtlety, internalStructure, calcification, sphericity, margin, lobulation, spiculation, texture = radiomic_features

    # Start with diameter
    description = f"A lung nodule with a diameter of {diameter:.2f} mm, located in the {anatomic_region}, "

    # Add subtlety
    subtlety_desc = {
        1: "extremely subtle",
        2: "moderately subtle",
        3: "fairly subtle",
        4: "moderately obvious",
        5: "obvious"
    }.get(subtlety, "unknown")
    description += f"{subtlety_desc}, "
    
    # Add internal structure
    internal_structure_desc = {
        1: "soft tissue composition",
        2: "fluid composition",
        3: "fat composition",
        4: "air composition"
    }.get(internalStructure, "unknown internal structure")
    description += f"with {internal_structure_desc}, "
    
    # Add calcification
    calc_desc = {
        1: "popcorn calcification",
        2: "laminated calcification",
        3: "solid calcification",
        4: "non-central calcification",
        5: "central calcification",
        6: "no calcification"
    }.get(calcification, "unknown")
    description += f"showing {calc_desc}, "
    
    # Add shape characteristics
    shape_desc = []
    if sphericity:
        sphericity_desc = {
            1: "linear",
            2: "ovoid or linear",
            3: "ovoid",
            4: "ovoid or round",
            5: "round"
        }.get(sphericity, "unknown")
        shape_desc.append(f"{sphericity_desc} shape")
    
    if margin:
        margin_desc = {
            1: "poorly defined",
            2: "near poorly defined",
            3: "medium",
            4: "near sharp",
            5: "sharp"
        }.get(margin, "unknown")
        shape_desc.append(f"{margin_desc} margins")
    
    if lobulation:
        lobulation_desc = {
            1: "no lobulation",
            2: "nearly no lobulation",
            3: "moderate lobulation",
            4: "near marked lobulation",
            5: "marked lobulation"
        }.get(lobulation, "unknown")
        shape_desc.append(f"{lobulation_desc}")
    
    if spiculation:
        spiculation_desc = {
            1: "no spiculation",
            2: "nearly no spiculation",
            3: "moderate spiculation",
            4: "near marked spiculation",
            5: "marked spiculation"
        }.get(spiculation, "unknown")
        shape_desc.append(f"{spiculation_desc}")
    
    if shape_desc:
        description += "with " + ", ".join(shape_desc) + ", "
    
    # Add texture
    if texture:
        texture_desc = {
            1: "non-solid or Ground-Glass Opacity",
            2: "non-solid or mixed",
            3: "part solid or mixed",
            4: "solid or mixed",
            5: "solid"
        }.get(texture, "unknown")
        description += f"and {texture_desc} texture."

    return description 