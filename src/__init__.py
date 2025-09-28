"""
CSV-XLSX Anomaly Detector Package
"""

__version__ = "1.0.0"
__author__ = "CSV-XLSX Anomaly Detector"
__description__ = "Python tool using CatBoost ML to compare CSV training data against XLSX files and detect anomalies"

from .data_processor import DataProcessor
from .model_trainer import ModelTrainer
from .anomaly_detector import AnomalyDetector
from .report_generator import ReportGenerator

__all__ = [
    'DataProcessor',
    'ModelTrainer', 
    'AnomalyDetector',
    'ReportGenerator'
]