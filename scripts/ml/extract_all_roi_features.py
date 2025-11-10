"""
Extract features from all 116 AAL ROIs
從所有 116 個 AAL 腦區提取特徵
"""

import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import datasets
from nilearn import image as nimg
from nilearn.maskers import NiftiLabelsMasker
import warnings
warnings.filterwarnings('ignore')

# AAL atlas 的所有 116 個 ROI
ALL_AAL_ROIS = {
    # Frontal Lobe (前額葉)
    'Precentral_L': 1, 'Precentral_R': 2,
    'Frontal_Sup_L': 3, 'Frontal_Sup_R': 4,
    'Frontal_Sup_Orb_L': 5, 'Frontal_Sup_Orb_R': 6,
    'Frontal_Mid_L': 7, 'Frontal_Mid_R': 8,
    'Frontal_Mid_Orb_L': 9, 'Frontal_Mid_Orb_R': 10,
    'Frontal_Inf_Oper_L': 11, 'Frontal_Inf_Oper_R': 12,
    'Frontal_Inf_Tri_L': 13, 'Frontal_Inf_Tri_R': 14,
    'Frontal_Inf_Orb_L': 15, 'Frontal_Inf_Orb_R': 16,
    'Rolandic_Oper_L': 17, 'Rolandic_Oper_R': 18,
    'Supp_Motor_Area_L': 19, 'Supp_Motor_Area_R': 20,
    'Olfactory_L': 21, 'Olfactory_R': 22,
    'Frontal_Sup_Medial_L': 23, 'Frontal_Sup_Medial_R': 24,
    'Frontal_Med_Orb_L': 25, 'Frontal_Med_Orb_R': 26,
    'Rectus_L': 27, 'Rectus_R': 28,
    
    # Insula & Cingulate (島葉與扣帶迴)
    'Insula_L': 29, 'Insula_R': 30,
    'Cingulum_Ant_L': 31, 'Cingulum_Ant_R': 32,
    'Cingulum_Mid_L': 33, 'Cingulum_Mid_R': 34,
    'Cingulum_Post_L': 35, 'Cingulum_Post_R': 36,
    
    # Hippocampus & Parahippocampal (海馬迴)
    'Hippocampus_L': 37, 'Hippocampus_R': 38,
    'ParaHippocampal_L': 39, 'ParaHippocampal_R': 40,
    'Amygdala_L': 41, 'Amygdala_R': 42,
    
    # Calcarine & Occipital (枕葉)
    'Calcarine_L': 43, 'Calcarine_R': 44,
    'Cuneus_L': 45, 'Cuneus_R': 46,
    'Lingual_L': 47, 'Lingual_R': 48,
    'Occipital_Sup_L': 49, 'Occipital_Sup_R': 50,
    'Occipital_Mid_L': 51, 'Occipital_Mid_R': 52,
    'Occipital_Inf_L': 53, 'Occipital_Inf_R': 54,
    'Fusiform_L': 55, 'Fusiform_R': 56,
    
    # Postcentral & Parietal (頂葉)
    'Postcentral_L': 57, 'Postcentral_R': 58,
    'Parietal_Sup_L': 59, 'Parietal_Sup_R': 60,
    'Parietal_Inf_L': 61, 'Parietal_Inf_R': 62,
    'SupraMarginal_L': 63, 'SupraMarginal_R': 64,
    'Angular_L': 65, 'Angular_R': 66,
    'Precuneus_L': 67, 'Precuneus_R': 68,
    'Paracentral_Lobule_L': 69, 'Paracentral_Lobule_R': 70,
    
    # Temporal Lobe (顳葉)
    'Heschl_L': 71, 'Heschl_R': 72,
    'Temporal_Sup_L': 73, 'Temporal_Sup_R': 74,
    'Temporal_Pole_Sup_L': 75, 'Temporal_Pole_Sup_R': 76,
    'Temporal_Mid_L': 77, 'Temporal_Mid_R': 78,
    'Temporal_Pole_Mid_L': 79, 'Temporal_Pole_Mid_R': 80,
    'Temporal_Inf_L': 81, 'Temporal_Inf_R': 82,
    
    # Subcortical (皮質下結構)
    'Caudate_L': 71, 'Caudate_R': 72,
    'Putamen_L': 73, 'Putamen_R': 74,
    'Pallidum_L': 75, 'Pallidum_R': 76,
    'Thalamus_L': 77, 'Thalamus_R': 78,
    
    # Cerebellum (小腦)
    'Cerebelum_Crus1_L': 91, 'Cerebelum_Crus1_R': 92,
    'Cerebelum_Crus2_L': 93, 'Cerebelum_Crus2_R': 94,
    'Cerebelum_3_L': 95, 'Cerebelum_3_R': 96,
    'Cerebelum_4_5_L': 97, 'Cerebelum_4_5_R': 98,
    'Cerebelum_6_L': 99, 'Cerebelum_6_R': 100,
    'Cerebelum_7b_L': 101, 'Cerebelum_7b_R': 102,
    'Cerebelum_8_L': 103, 'Cerebelum_8_R': 104,
    'Cerebelum_9_L': 105, 'Cerebelum_9_R': 106,
    'Cerebelum_10_L': 107, 'Cerebelum_10_R': 108,
    'Vermis_1_2': 109, 'Vermis_3': 110,
    'Vermis_4_5': 111, 'Vermis_6': 112,
    'Vermis_7': 113, 'Vermis_8': 114,
    'Vermis_9': 115, 'Vermis_10': 116
}


def load_aal_atlas():
    """載入 AAL atlas 並創建 masker"""
    print("Loading AAL atlas...")
    
    try:
        # Load AAL atlas using nilearn
        aal_atlas = datasets.fetch_atlas_aal(version='SPM12')
        aal_img = nimg.load_img(aal_atlas.maps)
        
        # Create masker for extracting ROI features
        masker = NiftiLabelsMasker(
            labels_img=aal_img,
            standardize=False,
            strategy='mean'  # Extract mean intensity per ROI
        )
        
        print(f"✓ Loaded AAL atlas from nilearn")
        print(f"✓ Atlas contains {len(aal_atlas.labels)} regions")
        
        return masker, aal_atlas.labels
        
    except Exception as e:
        print(f"⚠ Could not load AAL atlas: {e}")
        print("  Please check your nilearn installation")
        return None, None


def extract_all_subjects(data_dir, masker, roi_labels):
    """
    提取所有受試者的特徵
    
    Parameters:
    -----------
    data_dir : str
        數據目錄路徑
    masker : NiftiLabelsMasker
        AAL atlas masker
    roi_labels : list
        ROI 標籤列表
    
    Returns:
    --------
    features_df : DataFrame
        所有受試者的特徵矩陣
    """
    data_dir = Path(data_dir)
    
    all_features = []
    all_labels = []
    all_subjects = []
    
    # Process NC subjects
    nc_dir = data_dir / 'NC'
    if nc_dir.exists():
        nc_files = sorted(list(nc_dir.glob('*_T1.nii.gz')))
        print(f"\nProcessing {len(nc_files)} NC subjects...")
        
        for i, mri_file in enumerate(nc_files, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(nc_files)}")
            
            try:
                # Load MRI image
                mri_img = nimg.load_img(str(mri_file))
                
                # Extract ROI features using masker
                features = masker.fit_transform(mri_img)
                
                # features is a 2D array (1, n_rois), flatten it
                features = features.flatten()
                
                all_features.append(features)
                all_labels.append('NC')
                all_subjects.append(mri_file.stem.replace('_T1', ''))
                
            except Exception as e:
                print(f"  ⚠ Error processing {mri_file.name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Process AD subjects
    ad_dir = data_dir / 'AD'
    if ad_dir.exists():
        ad_files = sorted(list(ad_dir.glob('*_T1.nii.gz')))
        print(f"\nProcessing {len(ad_files)} AD subjects...")
        
        for i, mri_file in enumerate(ad_files, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(ad_files)}")
            
            try:
                # Load MRI image
                mri_img = nimg.load_img(str(mri_file))
                
                # Extract ROI features using masker
                features = masker.fit_transform(mri_img)
                
                # features is a 2D array (1, n_rois), flatten it
                features = features.flatten()
                
                all_features.append(features)
                all_labels.append('AD')
                all_subjects.append(mri_file.stem.replace('_T1', ''))
                
            except Exception as e:
                print(f"  ⚠ Error processing {mri_file.name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Create DataFrame
    features_array = np.array(all_features)
    
    # Create column names from ROI labels
    # Remove 'Background' if it exists (first label is usually background)
    roi_names = [label.decode('utf-8') if isinstance(label, bytes) else label 
                 for label in roi_labels]
    
    # The masker excludes background, so we need to skip the first label
    if len(roi_names) > features_array.shape[1]:
        roi_names = roi_names[1:]  # Skip 'Background'
    
    print(f"\n✓ Extracted {features_array.shape[1]} ROI features")
    print(f"  (AAL atlas has {len(roi_labels)} labels including background)")
    
    # Create DataFrame
    features_df = pd.DataFrame(features_array, columns=roi_names)
    features_df.insert(0, 'subject_id', all_subjects)
    features_df.insert(1, 'label', all_labels)
    features_df.insert(2, 'label_id', [0 if l == 'NC' else 1 for l in all_labels])
    
    return features_df


def main():
    """主函數"""
    print("="*80)
    print("Extract All AAL ROI Features")
    print("="*80)
    
    # Configuration
    data_dir = Path('E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI')
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load AAL atlas
        masker, roi_labels = load_aal_atlas()
        
        if masker is None:
            print("\n⚠ Cannot proceed without AAL atlas")
            print("Please check your nilearn installation")
            return
        
        # Extract features from all subjects
        print(f"\nExtracting features from: {data_dir}")
        features_df = extract_all_subjects(data_dir, masker, roi_labels)
        
        # Save results
        output_file = output_dir / 'all_aal_roi_features.csv'
        features_df.to_csv(output_file, index=False)
        
        print("\n" + "="*80)
        print("Feature Extraction Complete!")
        print("="*80)
        print(f"\nExtracted features: {len(roi_labels)} ROIs")
        print(f"Total subjects: {len(features_df)}")
        print(f"  NC: {(features_df['label'] == 'NC').sum()}")
        print(f"  AD: {(features_df['label'] == 'AD').sum()}")
        print(f"\nSaved to: {output_file}")
        
        # Show sample
        print("\nSample data (first 5 subjects, first 10 features):")
        print(features_df.iloc[:5, :13])
        
        # Show statistics
        print("\nFeature statistics (first 10 ROIs):")
        print(features_df.iloc[:, 3:13].describe())
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
