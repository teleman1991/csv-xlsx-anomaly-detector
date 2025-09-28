"""
Model Training Module
Handles CatBoost model training, validation, and persistence.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles CatBoost model training and evaluation."""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = None
        self.model_type = 'anomaly'  # or 'regression'
        
    def train(self, data, target_column=None, test_size=0.2, random_state=42):
        """Train CatBoost model on the data."""
        print("\n🤖 Starting model training...")
        
        # Prepare data for training
        processed_data = self._prepare_training_data(data, target_column)
        
        if processed_data.empty:
            raise ValueError("No valid data for training after preprocessing")
        
        # For anomaly detection, we'll create synthetic features
        if target_column is None:
            print("🎯 Training anomaly detection model...")
            return self._train_anomaly_model(processed_data, test_size, random_state)
        else:
            print(f"🎯 Training regression model with target: {target_column}")
            return self._train_regression_model(processed_data, target_column, test_size, random_state)
    
    def _prepare_training_data(self, data, target_column):
        """Prepare data for training by encoding categorical variables."""
        processed_data = data.copy()
        
        # Remove rows with all NaN values
        processed_data = processed_data.dropna(how='all')
        
        # Identify categorical and numerical columns
        categorical_columns = []
        numerical_columns = []
        
        for col in processed_data.columns:
            if col == target_column:
                continue
                
            # Check if column is numerical
            try:
                pd.to_numeric(processed_data[col], errors='raise')
                numerical_columns.append(col)
            except (ValueError, TypeError):
                categorical_columns.append(col)
        
        print(f"📊 Categorical columns: {categorical_columns}")
        print(f"📊 Numerical columns: {numerical_columns}")
        
        # Encode categorical variables
        for col in categorical_columns:
            if col in processed_data.columns:
                le = LabelEncoder()
                # Fill NaN with a placeholder
                processed_data[col] = processed_data[col].fillna('MISSING')
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
                self.label_encoders[col] = le
        
        # Handle numerical columns
        for col in numerical_columns:
            if col in processed_data.columns:
                # Convert to numeric, replacing non-numeric with NaN
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                # Fill NaN with median
                median_val = processed_data[col].median()
                processed_data[col] = processed_data[col].fillna(median_val)
        
        self.feature_columns = [col for col in processed_data.columns if col != target_column]
        
        return processed_data
    
    def _train_anomaly_model(self, data, test_size, random_state):
        """Train model for anomaly detection using reconstruction approach."""
        print("🔍 Creating features for anomaly detection...")
        
        # Create feature matrix
        X = data[self.feature_columns]
        
        # For anomaly detection, we'll train a model to predict each column from others
        # This creates a baseline for "normal" patterns
        models = {}
        
        for target_col in self.feature_columns:
            print(f"   Training model for column: {target_col}")
            
            # Features are all other columns
            feature_cols = [col for col in self.feature_columns if col != target_col]
            if not feature_cols:
                continue
                
            X_col = X[feature_cols]
            y_col = X[target_col]
            
            # Remove rows where target is NaN
            mask = ~y_col.isna()
            X_col_clean = X_col[mask]
            y_col_clean = y_col[mask]
            
            if len(X_col_clean) == 0:
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_col_clean, y_col_clean, test_size=test_size, random_state=random_state
            )
            
            # Determine if target is categorical or numerical
            if y_col_clean.dtype in ['int64', 'float64'] and len(y_col_clean.unique()) > 10:
                # Regression
                model = CatBoostRegressor(
                    iterations=100,
                    learning_rate=0.1,
                    depth=6,
                    verbose=False,
                    random_state=random_state
                )
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                score = mean_squared_error(y_test, y_pred)
                print(f"     MSE for {target_col}: {score:.4f}")
                
            else:
                # Classification
                model = CatBoostClassifier(
                    iterations=100,
                    learning_rate=0.1,
                    depth=6,
                    verbose=False,
                    random_state=random_state
                )
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                print(f"     Accuracy for {target_col}: {score:.4f}")
            
            models[target_col] = {
                'model': model,
                'feature_columns': feature_cols,
                'type': 'regression' if y_col_clean.dtype in ['int64', 'float64'] and len(y_col_clean.unique()) > 10 else 'classification'
            }
        
        self.model = models
        print(f"✅ Trained {len(models)} column prediction models")
        
        return models
    
    def _train_regression_model(self, data, target_column, test_size, random_state):
        """Train a single regression model."""
        X = data[self.feature_columns]
        y = data[target_column]
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            raise ValueError(f"No valid data for target column: {target_column}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        model = CatBoostRegressor(
            iterations=200,
            learning_rate=0.1,
            depth=8,
            verbose=False,
            random_state=random_state
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"✅ Model trained - MSE: {mse:.4f}")
        
        self.model = model
        self.target_column = target_column
        self.model_type = 'regression'
        
        return model
    
    def save_model(self, model, file_path):
        """Save model and encoders to file."""
        print(f"💾 Saving model to: {file_path}")
        
        model_data = {
            'model': model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_type': self.model_type
        }
        
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save using joblib for better compatibility
        joblib.dump(model_data, file_path)
        print("✅ Model saved successfully")
    
    def load_model(self, file_path):
        """Load model and encoders from file."""
        print(f"📦 Loading model from: {file_path}")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        model_data = joblib.load(file_path)
        
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data.get('target_column')
        self.model_type = model_data.get('model_type', 'anomaly')
        
        print("✅ Model loaded successfully")
        return self.model
    
    def predict(self, data):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Prepare data the same way as training
        processed_data = self._prepare_prediction_data(data)
        
        if self.model_type == 'anomaly':
            return self._predict_anomaly(processed_data)
        else:
            return self._predict_regression(processed_data)
    
    def _prepare_prediction_data(self, data):
        """Prepare data for prediction."""
        processed_data = data.copy()
        
        # Apply same encodings as training
        for col, encoder in self.label_encoders.items():
            if col in processed_data.columns:
                # Handle unseen categories
                processed_data[col] = processed_data[col].fillna('MISSING')
                processed_data[col] = processed_data[col].astype(str)
                
                # Transform known categories, assign a default for unknown
                known_classes = set(encoder.classes_)
                mask = processed_data[col].isin(known_classes)
                
                # For unknown categories, assign the most frequent class
                unknown_replacement = encoder.classes_[0] if len(encoder.classes_) > 0 else 'MISSING'
                processed_data.loc[~mask, col] = unknown_replacement
                
                processed_data[col] = encoder.transform(processed_data[col])
        
        # Handle numerical columns
        for col in self.feature_columns:
            if col in processed_data.columns and col not in self.label_encoders:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                processed_data[col] = processed_data[col].fillna(processed_data[col].median())
        
        return processed_data
    
    def _predict_anomaly(self, data):
        """Make anomaly predictions."""
        predictions = {}
        
        for target_col, model_info in self.model.items():
            if target_col in data.columns:
                feature_cols = model_info['feature_columns']
                model = model_info['model']
                
                # Get features that exist in the data
                available_features = [col for col in feature_cols if col in data.columns]
                if available_features:
                    X = data[available_features]
                    pred = model.predict(X)
                    predictions[target_col] = pred
        
        return predictions
    
    def _predict_regression(self, data):
        """Make regression predictions."""
        available_features = [col for col in self.feature_columns if col in data.columns]
        if not available_features:
            return np.array([])
            
        X = data[available_features]
        return self.model.predict(X)