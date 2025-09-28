"""
Data Processing Module
Handles loading, cleaning, and aligning CSV and XLSX data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data loading and preprocessing for CSV and XLSX files."""
    
    def __init__(self):
        self.csv_columns = None
        self.csv_dtypes = None
        
    def load_csv(self, file_path):
        """Load and validate CSV training data."""
        logger.info(f"Loading CSV file: {file_path}")
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            data = None
            
            for encoding in encodings:
                try:
                    data = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if data is None:
                raise ValueError("Could not read CSV file with any encoding")
            
            self.csv_columns = data.columns.tolist()
            
            # Basic validation
            if data.empty:
                raise ValueError("CSV file is empty")
                
            print(f"✅ Loaded CSV: {len(data)} rows × {len(data.columns)} columns")
            print(f"📊 Columns: {list(data.columns)}")
            
            # Show data types and sample
            print("\n📋 Data Preview:")
            print(data.head(3).to_string())
            print(f"\n📈 Data Types:")
            print(data.dtypes.to_string())
            
            # Store original data types for comparison
            self.csv_dtypes = data.dtypes.to_dict()
            
            return self._clean_data(data)
            
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise
    
    def load_xlsx(self, file_path):
        """Load XLSX test data."""
        logger.info(f"Loading XLSX file: {file_path}")
        
        try:
            # Try to read Excel file
            data = pd.read_excel(file_path, engine='openpyxl')
            
            if data.empty:
                raise ValueError("XLSX file is empty")
                
            print(f"✅ Loaded XLSX: {len(data)} rows × {len(data.columns)} columns")
            print(f"📊 Columns: {list(data.columns)}")
            
            # Show preview
            print("\n📋 Data Preview:")
            print(data.head(3).to_string())
            
            return self._clean_data(data)
            
        except Exception as e:
            logger.error(f"Error loading XLSX: {str(e)}")
            raise
    
    def align_columns(self, csv_data, xlsx_data):
        """Align XLSX columns to match CSV structure."""
        if self.csv_columns is None:
            raise ValueError("CSV data must be loaded first")
            
        print("\n🔄 Aligning columns...")
        
        # Find matching columns (case-insensitive)
        csv_cols_lower = [col.lower().strip() for col in self.csv_columns]
        xlsx_cols_lower = [col.lower().strip() for col in xlsx_data.columns]
        
        # Create mapping
        column_mapping = {}
        for csv_col, csv_lower in zip(self.csv_columns, csv_cols_lower):
            for xlsx_col, xlsx_lower in zip(xlsx_data.columns, xlsx_cols_lower):
                if csv_lower == xlsx_lower:
                    column_mapping[xlsx_col] = csv_col
                    break
        
        matching_columns = list(column_mapping.keys())
        missing_columns = [col for col in self.csv_columns if col not in column_mapping.values()]
        
        if missing_columns:
            print(f"⚠️  Missing columns in XLSX: {missing_columns}")
            
        if not matching_columns:
            raise ValueError("No matching columns found between CSV and XLSX")
            
        print(f"✅ Matching columns ({len(matching_columns)}): {list(column_mapping.values())}")
        
        # Rename XLSX columns to match CSV
        aligned_data = xlsx_data[matching_columns].copy()
        aligned_data.columns = [column_mapping[col] for col in matching_columns]
        
        # Reorder to match CSV column order
        csv_order = [col for col in self.csv_columns if col in aligned_data.columns]
        aligned_data = aligned_data[csv_order]
        
        return aligned_data
    
    def _clean_data(self, data):
        """Basic data cleaning and standardization."""
        # Make a copy to avoid modifying original
        cleaned_data = data.copy()
        
        # Handle common null representations
        null_values = ['nan', 'NaN', 'None', 'null', '', 'NULL', 'Null', '#N/A', 'N/A']
        
        for col in cleaned_data.columns:
            # Replace null values
            cleaned_data[col] = cleaned_data[col].replace(null_values, np.nan)
            
            # Strip whitespace for string columns
            if cleaned_data[col].dtype == 'object':
                cleaned_data[col] = cleaned_data[col].astype(str).str.strip()
                # Replace 'nan' strings that come from conversion
                cleaned_data[col] = cleaned_data[col].replace('nan', np.nan)
        
        return cleaned_data
    
    def get_data_summary(self, data, name="Data"):
        """Generate summary statistics for data."""
        summary = {
            'name': name,
            'rows': len(data),
            'columns': len(data.columns),
            'missing_values': data.isnull().sum().sum(),
            'column_types': data.dtypes.to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum()
        }
        return summary