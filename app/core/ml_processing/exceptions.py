"""
Custom exceptions for ML integration
"""


class MLIntegrationError(Exception):
    """Base exception for ML integration errors"""
    pass


class ModelLoadError(MLIntegrationError):
    """Raised when model loading fails"""
    pass


class FeatureExtractionError(MLIntegrationError):
    """Raised when feature extraction fails"""
    pass


class AtlasLoadError(MLIntegrationError):
    """Raised when atlas loading fails"""
    pass


class PredictionError(MLIntegrationError):
    """Raised when prediction fails"""
    pass
