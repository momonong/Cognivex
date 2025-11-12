# Test Fixtures

This directory contains mock data files for testing.

## Files

- `mock_t1_mri.nii.gz`: Mock T1 MRI file for testing feature extraction
- `mock_model.pkl`: Mock Random Forest model for testing
- `mock_scaler.pkl`: Mock StandardScaler for testing

## Note

These files are not included in the repository. To run tests that require them:

1. Either use real model files from `model/ml/final/`
2. Or create mock files using the provided scripts
3. Tests will automatically skip if files are not available
