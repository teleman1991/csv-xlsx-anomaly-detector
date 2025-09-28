"""
Report Generation Module
Creates detailed reports of detected differences and anomalies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates comprehensive reports of detected differences and anomalies."""
    
    def __init__(self):
        self.report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_report(self, differences, output_dir, csv_name="training", xlsx_name="test"):
        """Generate comprehensive difference reports in multiple formats."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 Generating reports in: {output_path}")
        
        # Convert differences to DataFrame
        if differences:
            df = pd.DataFrame(differences)
        else:
            df = pd.DataFrame(columns=['row_index', 'column', 'test_value', 'expected_pattern', 
                                     'difference_type', 'severity', 'anomaly_score'])
        
        # Generate different report formats
        report_paths = []
        
        # 1. Detailed CSV Report
        csv_path = self._generate_csv_report(df, output_path, csv_name, xlsx_name)
        report_paths.append(csv_path)
        
        # 2. Summary Excel Report
        excel_path = self._generate_excel_report(df, output_path, csv_name, xlsx_name)
        report_paths.append(excel_path)
        
        # 3. HTML Dashboard Report
        html_path = self._generate_html_report(df, output_path, csv_name, xlsx_name)
        report_paths.append(html_path)
        
        return report_paths
    
    def _generate_csv_report(self, df, output_path, csv_name, xlsx_name):
        """Generate detailed CSV report."""
        filename = f"differences_detailed_{self.report_timestamp}.csv"
        file_path = output_path / filename
        
        if not df.empty:
            # Sort by row index and severity
            severity_order = {'high': 3, 'medium': 2, 'low': 1}
            df['severity_score'] = df['severity'].map(severity_order)
            df_sorted = df.sort_values(['row_index', 'severity_score'], ascending=[True, False])
            df_sorted = df_sorted.drop('severity_score', axis=1)
            
            # Add additional columns for context
            df_sorted.insert(0, 'report_timestamp', self.report_timestamp)
            df_sorted.insert(1, 'training_file', csv_name)
            df_sorted.insert(2, 'test_file', xlsx_name)
            
            df_sorted.to_csv(file_path, index=False)
        else:
            # Create empty report with headers
            empty_df = pd.DataFrame(columns=['report_timestamp', 'training_file', 'test_file', 
                                           'row_index', 'column', 'test_value', 'expected_pattern',
                                           'difference_type', 'severity', 'anomaly_score'])
            empty_df.to_csv(file_path, index=False)
        
        print(f"   ✅ CSV report: {filename}")
        return file_path
    
    def _generate_excel_report(self, df, output_path, csv_name, xlsx_name):
        """Generate comprehensive Excel report with multiple sheets."""
        filename = f"differences_summary_{self.report_timestamp}.xlsx"
        file_path = output_path / filename
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = self._create_summary(df, csv_name, xlsx_name)
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed differences sheet
            if not df.empty:
                df_copy = df.copy()
                df_copy.insert(0, 'training_file', csv_name)
                df_copy.insert(1, 'test_file', xlsx_name)
                df_copy.to_excel(writer, sheet_name='Detailed_Differences', index=False)
            
            # By column analysis
            if not df.empty:
                column_analysis = self._analyze_by_column(df)
                column_analysis.to_excel(writer, sheet_name='By_Column_Analysis', index=False)
            
            # By type analysis
            if not df.empty:
                type_analysis = self._analyze_by_type(df)
                type_analysis.to_excel(writer, sheet_name='By_Type_Analysis', index=False)
            
            # High severity only
            if not df.empty:
                high_severity = df[df['severity'] == 'high'].copy()
                if not high_severity.empty:
                    high_severity.to_excel(writer, sheet_name='High_Severity_Only', index=False)
        
        print(f"   ✅ Excel report: {filename}")
        return file_path
    
    def _generate_html_report(self, df, output_path, csv_name, xlsx_name):
        """Generate interactive HTML dashboard report."""
        filename = f"differences_dashboard_{self.report_timestamp}.html"
        file_path = output_path / filename
        
        # Create HTML content
        html_content = self._create_html_dashboard(df, csv_name, xlsx_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"   ✅ HTML dashboard: {filename}")
        return file_path
    
    def _create_summary(self, df, csv_name, xlsx_name):
        """Create summary statistics."""
        if df.empty:
            return {
                'report_timestamp': self.report_timestamp,
                'training_file': csv_name,
                'test_file': xlsx_name,
                'total_differences': 0,
                'affected_rows': 0,
                'affected_columns': 0,
                'high_severity_count': 0,
                'medium_severity_count': 0,
                'low_severity_count': 0,
                'avg_anomaly_score': 0,
                'max_anomaly_score': 0
            }
        
        return {
            'report_timestamp': self.report_timestamp,
            'training_file': csv_name,
            'test_file': xlsx_name,
            'total_differences': len(df),
            'affected_rows': df['row_index'].nunique(),
            'affected_columns': df['column'].nunique(),
            'high_severity_count': len(df[df['severity'] == 'high']),
            'medium_severity_count': len(df[df['severity'] == 'medium']),
            'low_severity_count': len(df[df['severity'] == 'low']),
            'avg_anomaly_score': df['anomaly_score'].mean(),
            'max_anomaly_score': df['anomaly_score'].max()
        }
    
    def _analyze_by_column(self, df):
        """Analyze differences by column."""
        if df.empty:
            return pd.DataFrame()
        
        column_stats = df.groupby('column').agg({
            'row_index': 'count',
            'anomaly_score': ['mean', 'max'],
            'severity': lambda x: x.value_counts().to_dict()
        }).reset_index()
        
        # Flatten column names
        column_stats.columns = ['column', 'difference_count', 'avg_anomaly_score', 
                               'max_anomaly_score', 'severity_breakdown']
        
        return column_stats.sort_values('difference_count', ascending=False)
    
    def _analyze_by_type(self, df):
        """Analyze differences by type."""
        if df.empty:
            return pd.DataFrame()
        
        type_stats = df.groupby('difference_type').agg({
            'row_index': 'count',
            'anomaly_score': ['mean', 'max'],
            'column': lambda x: x.nunique()
        }).reset_index()
        
        # Flatten column names
        type_stats.columns = ['difference_type', 'occurrence_count', 'avg_anomaly_score',
                             'max_anomaly_score', 'affected_columns']
        
        return type_stats.sort_values('occurrence_count', ascending=False)
    
    def _create_html_dashboard(self, df, csv_name, xlsx_name):
        """Create HTML dashboard with charts and tables."""
        summary = self._create_summary(df, csv_name, xlsx_name)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Comparison Report - {self.report_timestamp}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .summary-card p {{
            margin: 0;
            opacity: 0.9;
        }}
        .severity-high {{
            background: linear-gradient(135deg, #dc3545, #c82333);
        }}
        .severity-medium {{
            background: linear-gradient(135deg, #ffc107, #e0a800);
        }}
        .severity-low {{
            background: linear-gradient(135deg, #28a745, #1e7e34);
        }}
        .table-container {{
            margin: 30px 0;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #007bff;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .severity-badge {{
            padding: 4px 8px;
            border-radius: 4px;
            color: white;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .badge-high {{
            background-color: #dc3545;
        }}
        .badge-medium {{
            background-color: #ffc107;
            color: #333;
        }}
        .badge-low {{
            background-color: #28a745;
        }}
        .no-data {{
            text-align: center;
            color: #28a745;
            font-size: 1.2em;
            margin: 40px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Data Comparison Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Training Data:</strong> {csv_name} | <strong>Test Data:</strong> {xlsx_name}</p>
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>{summary['total_differences']}</h3>
                <p>Total Differences</p>
            </div>
            <div class="summary-card">
                <h3>{summary['affected_rows']}</h3>
                <p>Affected Rows</p>
            </div>
            <div class="summary-card">
                <h3>{summary['affected_columns']}</h3>
                <p>Affected Columns</p>
            </div>
            <div class="summary-card severity-high">
                <h3>{summary['high_severity_count']}</h3>
                <p>High Severity</p>
            </div>
            <div class="summary-card severity-medium">
                <h3>{summary['medium_severity_count']}</h3>
                <p>Medium Severity</p>
            </div>
            <div class="summary-card severity-low">
                <h3>{summary['low_severity_count']}</h3>
                <p>Low Severity</p>
            </div>
        </div>
"""
        
        if df.empty:
            html_template += """
        <div class="no-data">
            <h2>🎉 No Differences Found!</h2>
            <p>The test data matches the training data patterns perfectly.</p>
        </div>
"""
        else:
            # Add detailed differences table
            html_template += f"""
        <div class="table-container">
            <h2>📋 Detailed Differences</h2>
            <table>
                <thead>
                    <tr>
                        <th>Row</th>
                        <th>Column</th>
                        <th>Test Value</th>
                        <th>Expected Pattern</th>
                        <th>Type</th>
                        <th>Severity</th>
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            # Add table rows (limit to first 100 for performance)
            for _, row in df.head(100).iterrows():
                severity_class = f"badge-{row['severity']}"
                html_template += f"""
                    <tr>
                        <td>{row['row_index']}</td>
                        <td>{row['column']}</td>
                        <td>{str(row['test_value'])[:50]}</td>
                        <td>{str(row['expected_pattern'])[:50]}</td>
                        <td>{row['difference_type']}</td>
                        <td><span class="severity-badge {severity_class}">{row['severity'].upper()}</span></td>
                        <td>{row['anomaly_score']:.3f}</td>
                    </tr>
"""
            
            html_template += """
                </tbody>
            </table>
        </div>
"""
            
            if len(df) > 100:
                html_template += f"""
        <p><em>Showing first 100 differences. Total: {len(df)} differences found.</em></p>
"""
        
        html_template += """
    </div>
</body>
</html>
"""
        
        return html_template