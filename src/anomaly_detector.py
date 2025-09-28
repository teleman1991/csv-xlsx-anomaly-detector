"""
Anomaly Detection Module
Compares XLSX data against trained model to detect differences and anomalies.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Detects anomalies and differences between training data patterns and test data."""
    
    def __init__(self, model, training_data):
        self.model = model
        self.training_data = training_data
        self.model_trainer = None  # Will be set if needed
        
    def compare_data(self, test_data, threshold=0.1):
        """Compare test data against training patterns and return differences."""
        print("\n🔍 Starting anomaly detection...")
        
        differences = []
        
        # Process row by row
        for idx, row in test_data.iterrows():
            row_differences = self._analyze_row(row, idx, threshold)
            differences.extend(row_differences)
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(test_data)} rows...")
        
        print(f"✅ Analysis complete - Found {len(differences)} differences")
        return differences
    
    def _analyze_row(self, row, row_index, threshold):
        """Analyze a single row for anomalies and differences."""
        row_differences = []
        
        # Convert row to DataFrame for model prediction
        row_df = pd.DataFrame([row])
        
        # 1. Direct value comparison with training data patterns
        direct_diffs = self._check_direct_differences(row, row_index)
        row_differences.extend(direct_diffs)
        
        # 2. ML-based anomaly detection
        ml_diffs = self._check_ml_anomalies(row_df, row_index, threshold)
        row_differences.extend(ml_diffs)
        
        # 3. Statistical outlier detection
        stat_diffs = self._check_statistical_outliers(row, row_index)
        row_differences.extend(stat_diffs)
        
        return row_differences
    
    def _check_direct_differences(self, row, row_index):
        """Check for direct value differences against training data."""
        differences = []
        
        for col in row.index:
            if col not in self.training_data.columns:
                continue
                
            test_value = row[col]
            training_values = self.training_data[col].dropna()
            
            # Skip if test value is NaN
            if pd.isna(test_value):
                # Check if NaN is common in training data
                nan_ratio = self.training_data[col].isna().sum() / len(self.training_data)
                if nan_ratio < 0.1:  # Less than 10% NaN in training
                    differences.append({
                        'row_index': row_index,
                        'column': col,
                        'test_value': test_value,
                        'expected_pattern': 'Non-null value',
                        'difference_type': 'unexpected_null',
                        'severity': 'medium',
                        'anomaly_score': 1.0 - nan_ratio
                    })
                continue
            
            # Check if value exists in training data
            if len(training_values) > 0:
                # For categorical data
                if training_values.dtype == 'object' or len(training_values.unique()) < 20:
                    unique_values = set(training_values.unique())
                    if str(test_value) not in [str(v) for v in unique_values]:
                        differences.append({
                            'row_index': row_index,
                            'column': col,
                            'test_value': test_value,
                            'expected_pattern': f'One of: {list(unique_values)[:10]}',
                            'difference_type': 'unexpected_category',
                            'severity': 'high',
                            'anomaly_score': 1.0
                        })
                
                # For numerical data
                else:
                    try:
                        test_num = float(test_value)
                        min_val = training_values.min()
                        max_val = training_values.max()
                        mean_val = training_values.mean()
                        std_val = training_values.std()
                        
                        # Check if value is outside reasonable range
                        if test_num < min_val or test_num > max_val:
                            differences.append({
                                'row_index': row_index,
                                'column': col,
                                'test_value': test_value,
                                'expected_pattern': f'Between {min_val:.2f} and {max_val:.2f}',
                                'difference_type': 'out_of_range',
                                'severity': 'high',
                                'anomaly_score': min(abs(test_num - min_val) / (max_val - min_val), 1.0) if max_val != min_val else 1.0
                            })
                        
                        # Check if value is statistical outlier (> 3 standard deviations)
                        elif std_val > 0:
                            z_score = abs(test_num - mean_val) / std_val
                            if z_score > 3:
                                differences.append({
                                    'row_index': row_index,
                                    'column': col,
                                    'test_value': test_value,
                                    'expected_pattern': f'Near {mean_val:.2f} ± {std_val:.2f}',
                                    'difference_type': 'statistical_outlier',
                                    'severity': 'medium',
                                    'anomaly_score': min(z_score / 3.0, 1.0)
                                })
                                
                    except (ValueError, TypeError):
                        # Value couldn't be converted to number
                        differences.append({
                            'row_index': row_index,
                            'column': col,
                            'test_value': test_value,
                            'expected_pattern': 'Numerical value',
                            'difference_type': 'type_mismatch',
                            'severity': 'high',
                            'anomaly_score': 1.0
                        })
        
        return differences
    
    def _check_ml_anomalies(self, row_df, row_index, threshold):
        """Use trained ML model to detect anomalies."""
        differences = []
        
        if self.model is None or not isinstance(self.model, dict):
            return differences
        
        try:
            # For each column model, predict the expected value
            for target_col, model_info in self.model.items():
                if target_col not in row_df.columns:
                    continue
                
                model = model_info['model']
                feature_cols = model_info['feature_columns']
                model_type = model_info['type']
                
                # Get available features
                available_features = [col for col in feature_cols if col in row_df.columns]
                if not available_features:
                    continue
                
                # Prepare features
                X = row_df[available_features]
                actual_value = row_df[target_col].iloc[0]
                
                # Skip if actual value is NaN
                if pd.isna(actual_value):
                    continue
                
                # Make prediction
                try:
                    predicted_value = model.predict(X)[0]
                    
                    # Calculate anomaly score based on model type
                    if model_type == 'regression':
                        # For regression, calculate relative error
                        if predicted_value != 0:
                            relative_error = abs(actual_value - predicted_value) / abs(predicted_value)
                        else:
                            relative_error = abs(actual_value - predicted_value)
                        
                        if relative_error > threshold:
                            differences.append({
                                'row_index': row_index,
                                'column': target_col,
                                'test_value': actual_value,
                                'expected_pattern': f'≈ {predicted_value:.4f}',
                                'difference_type': 'ml_anomaly_regression',
                                'severity': 'medium' if relative_error < 0.5 else 'high',
                                'anomaly_score': min(relative_error, 1.0)
                            })
                    
                    else:  # Classification
                        # For classification, check if prediction matches
                        if actual_value != predicted_value:
                            # Get prediction probability if available
                            try:
                                proba = model.predict_proba(X)[0]
                                confidence = max(proba) if len(proba) > 0 else 0.5
                            except:
                                confidence = 0.5
                            
                            differences.append({
                                'row_index': row_index,
                                'column': target_col,
                                'test_value': actual_value,
                                'expected_pattern': f'{predicted_value}',
                                'difference_type': 'ml_anomaly_classification',
                                'severity': 'high' if confidence > 0.8 else 'medium',
                                'anomaly_score': confidence
                            })
                
                except Exception as e:
                    logger.warning(f"Error in ML prediction for column {target_col}: {str(e)}")
                    continue
        
        except Exception as e:
            logger.warning(f"Error in ML anomaly detection: {str(e)}")
        
        return differences
    
    def _check_statistical_outliers(self, row, row_index):
        """Check for statistical outliers using IQR method."""
        differences = []
        
        for col in row.index:
            if col not in self.training_data.columns:
                continue
            
            test_value = row[col]
            training_values = self.training_data[col].dropna()
            
            # Skip non-numeric columns or if not enough data
            if len(training_values) < 10:
                continue
            
            try:
                test_num = float(test_value)
                training_numeric = pd.to_numeric(training_values, errors='coerce').dropna()
                
                if len(training_numeric) < 10:
                    continue
                
                # Calculate IQR
                Q1 = training_numeric.quantile(0.25)
                Q3 = training_numeric.quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Check if test value is an outlier
                if test_num < lower_bound or test_num > upper_bound:
                    # Calculate how far outside the bounds
                    if test_num < lower_bound:
                        distance = lower_bound - test_num
                    else:
                        distance = test_num - upper_bound
                    
                    # Normalize distance
                    anomaly_score = min(distance / (IQR if IQR > 0 else 1), 1.0)
                    
                    differences.append({
                        'row_index': row_index,
                        'column': col,
                        'test_value': test_value,
                        'expected_pattern': f'Between {lower_bound:.2f} and {upper_bound:.2f} (IQR)',
                        'difference_type': 'iqr_outlier',
                        'severity': 'low',
                        'anomaly_score': anomaly_score
                    })
            
            except (ValueError, TypeError):
                continue
        
        return differences
    
    def get_summary_stats(self, differences):
        """Generate summary statistics for detected differences."""
        if not differences:
            return {
                'total_differences': 0,
                'by_type': {},
                'by_severity': {},
                'by_column': {},
                'affected_rows': 0
            }
        
        df = pd.DataFrame(differences)
        
        summary = {
            'total_differences': len(differences),
            'by_type': df['difference_type'].value_counts().to_dict(),
            'by_severity': df['severity'].value_counts().to_dict(),
            'by_column': df['column'].value_counts().to_dict(),
            'affected_rows': df['row_index'].nunique(),
            'avg_anomaly_score': df['anomaly_score'].mean(),
            'max_anomaly_score': df['anomaly_score'].max()
        }
        
        return summary