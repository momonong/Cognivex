"""
檔案處理服務
處理 DICOM 到 NIfTI 轉換、元數據提取和檔案驗證
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


class FileProcessor:
    """檔案處理器類"""
    
    def __init__(self):
        self.supported_formats = {
            '.nii': 'nifti',
            '.nii.gz': 'nifti',
            '.dcm': 'dicom',
            '.json': 'metadata'
        }
    
    def validate_file_format(self, filename: str) -> bool:
        """驗證檔案格式是否支援"""
        return any(filename.lower().endswith(ext) for ext in self.supported_formats.keys())
    
    def get_file_type(self, filename: str) -> str:
        """根據檔案名判斷檔案類型"""
        filename_lower = filename.lower()
        for ext, file_type in self.supported_formats.items():
            if filename_lower.endswith(ext):
                return file_type
        return 'unknown'
    
    def extract_nifti_metadata(self, file_path: str) -> Dict[str, Any]:
        """提取 NIfTI 檔案元數據"""
        try:
            img = nib.load(file_path)
            header = img.header
            
            metadata = {
                'dimensions': [int(x) for x in img.shape],
                'voxel_size': [float(x) for x in header.get_zooms()],
                'data_type': str(img.get_data_dtype()),
                'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
                'orientation': str(nib.aff2axcodes(img.affine)),
                'units': {
                    'spatial': str(header.get_xyzt_units()[0]),
                    'temporal': str(header.get_xyzt_units()[1])
                }
            }
            
            # 檢查是否為 4D fMRI 數據
            if len(img.shape) == 4:
                metadata['is_4d'] = True
                metadata['time_points'] = int(img.shape[3])
                metadata['tr'] = float(header.get_zooms()[3]) if len(header.get_zooms()) > 3 else None
            else:
                metadata['is_4d'] = False
            
            # 檢查數據範圍
            try:
                data = img.get_fdata()
                metadata['data_range'] = {
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data))
                }
            except Exception as e:
                logger.warning(f"無法計算數據統計: {e}")
                metadata['data_range'] = None
            
            return metadata
            
        except Exception as e:
            logger.error(f"提取 NIfTI 元數據失敗: {e}")
            return {'error': str(e)}
    
    def extract_dicom_metadata(self, file_path: str) -> Dict[str, Any]:
        """提取 DICOM 檔案元數據"""
        try:
            # 這裡需要 pydicom 庫
            # 暫時返回基本資訊
            metadata = {
                'file_type': 'dicom',
                'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
                'note': 'DICOM 元數據提取需要 pydicom 庫'
            }
            
            # TODO: 實作完整的 DICOM 元數據提取
            # import pydicom
            # ds = pydicom.dcmread(file_path)
            # metadata.update({
            #     'patient_id': ds.get('PatientID', ''),
            #     'study_date': ds.get('StudyDate', ''),
            #     'modality': ds.get('Modality', ''),
            #     'manufacturer': ds.get('Manufacturer', ''),
            #     'scanner_model': ds.get('ManufacturerModelName', ''),
            #     'magnetic_field_strength': ds.get('MagneticFieldStrength', ''),
            #     'slice_thickness': ds.get('SliceThickness', ''),
            #     'repetition_time': ds.get('RepetitionTime', ''),
            #     'echo_time': ds.get('EchoTime', '')
            # })
            
            return metadata
            
        except Exception as e:
            logger.error(f"提取 DICOM 元數據失敗: {e}")
            return {'error': str(e)}
    
    def extract_json_metadata(self, file_path: str) -> Dict[str, Any]:
        """提取 JSON 元數據檔案內容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            metadata = {
                'file_type': 'metadata',
                'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
                'content': json_data,
                'keys': list(json_data.keys()) if isinstance(json_data, dict) else None
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"提取 JSON 元數據失敗: {e}")
            return {'error': str(e)}
    
    def extract_metadata(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """根據檔案類型提取元數據"""
        if not os.path.exists(file_path):
            return {'error': '檔案不存在'}
        
        if file_type == 'nifti':
            return self.extract_nifti_metadata(file_path)
        elif file_type == 'dicom':
            return self.extract_dicom_metadata(file_path)
        elif file_type == 'metadata':
            return self.extract_json_metadata(file_path)
        else:
            return {'error': f'不支援的檔案類型: {file_type}'}
    
    def validate_nifti_file(self, file_path: str) -> Dict[str, Any]:
        """驗證 NIfTI 檔案完整性"""
        validation_result = {
            'is_valid': True,
            'checks': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            # 檢查檔案是否存在
            if not os.path.exists(file_path):
                validation_result['is_valid'] = False
                validation_result['errors'].append('檔案不存在')
                return validation_result
            
            validation_result['checks'].append('檔案存在性檢查')
            
            # 檢查檔案大小
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                validation_result['is_valid'] = False
                validation_result['errors'].append('檔案大小為 0')
                return validation_result
            
            validation_result['checks'].append('檔案大小檢查')
            
            # 嘗試載入 NIfTI 檔案
            img = nib.load(file_path)
            validation_result['checks'].append('NIfTI 格式驗證')
            
            # 檢查影像維度
            shape = img.shape
            if len(shape) < 3:
                validation_result['warnings'].append(f'影像維度可能不正確: {shape}')
            elif len(shape) == 3:
                validation_result['checks'].append('3D 結構影像格式')
            elif len(shape) == 4:
                validation_result['checks'].append('4D 功能影像格式')
                if shape[3] < 10:
                    validation_result['warnings'].append(f'時間點數量較少: {shape[3]}')
            
            # 檢查體素大小
            voxel_sizes = img.header.get_zooms()
            if any(size <= 0 for size in voxel_sizes[:3]):
                validation_result['warnings'].append('體素大小異常')
            
            # 檢查數據類型
            dtype = img.get_data_dtype()
            if dtype not in [np.float32, np.float64, np.int16, np.int32]:
                validation_result['warnings'].append(f'數據類型可能不標準: {dtype}')
            
            validation_result['metadata'] = {
                'shape': [int(x) for x in shape],
                'voxel_size': [float(x) for x in voxel_sizes],
                'data_type': str(dtype),
                'file_size_mb': round(file_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'NIfTI 檔案驗證失敗: {str(e)}')
        
        return validation_result
    
    def convert_dicom_to_nifti(self, dicom_path: str, output_path: str) -> Dict[str, Any]:
        """將 DICOM 檔案轉換為 NIfTI 格式"""
        # 這個功能需要 dcm2niix 或 pydicom + nibabel
        # 暫時返回模擬結果
        
        result = {
            'success': False,
            'output_path': output_path,
            'message': 'DICOM 到 NIfTI 轉換功能待實作',
            'note': '需要安裝 dcm2niix 或實作 pydicom 轉換邏輯'
        }
        
        # TODO: 實作實際的轉換邏輯
        # 可以使用以下方法之一:
        # 1. 使用 dcm2niix 命令行工具
        # 2. 使用 pydicom + nibabel 進行轉換
        # 3. 使用 SimpleITK 進行轉換
        
        try:
            # 檢查輸入檔案
            if not os.path.exists(dicom_path):
                result['message'] = 'DICOM 檔案不存在'
                return result
            
            # 創建輸出目錄
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 暫時複製檔案作為佔位符
            # 實際實作時應該進行真正的格式轉換
            import shutil
            shutil.copy2(dicom_path, output_path + '.placeholder')
            
            result.update({
                'success': True,
                'message': '轉換完成 (模擬)',
                'output_path': output_path + '.placeholder'
            })
            
        except Exception as e:
            result['message'] = f'轉換失敗: {str(e)}'
        
        return result
    
    def get_preprocessing_recommendations(self, metadata: Dict[str, Any]) -> List[str]:
        """根據檔案元數據提供預處理建議"""
        recommendations = []
        
        if 'dimensions' in metadata:
            dims = metadata['dimensions']
            
            # 檢查是否為 4D fMRI 數據
            if len(dims) == 4:
                if dims[3] < 100:
                    recommendations.append('時間點數量較少，建議檢查掃描參數')
                
                if 'tr' in metadata and metadata['tr']:
                    tr = metadata['tr']
                    if tr < 1.0:
                        recommendations.append('TR 較短，適合高時間解析度分析')
                    elif tr > 3.0:
                        recommendations.append('TR 較長，建議檢查掃描協議')
            
            # 檢查空間解析度
            if 'voxel_size' in metadata:
                voxel_sizes = metadata['voxel_size'][:3]
                if any(size > 4.0 for size in voxel_sizes):
                    recommendations.append('空間解析度較低，可能影響分析精度')
                elif all(size < 2.0 for size in voxel_sizes):
                    recommendations.append('高空間解析度，適合精細結構分析')
        
        # 檢查數據範圍
        if 'data_range' in metadata and metadata['data_range']:
            data_range = metadata['data_range']
            if data_range['max'] - data_range['min'] < 100:
                recommendations.append('信號動態範圍較小，建議檢查數據品質')
        
        if not recommendations:
            recommendations.append('數據格式正常，可以進行標準預處理')
        
        return recommendations


# 全域檔案處理器實例
file_processor = FileProcessor()