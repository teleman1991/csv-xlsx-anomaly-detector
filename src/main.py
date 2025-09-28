#!/usr/bin/env python3
"""
CSV-XLSX Anomaly Detector
Main entry point for the data comparison tool.
"""

import click
import logging
import os
from pathlib import Path
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from anomaly_detector import AnomalyDetector
from report_generator import ReportGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def select_file(file_type, directory="data"):
    """Interactive file selection helper"""
    data_dir = Path(directory)
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        
    if file_type == "csv":
        files = list(data_dir.glob("*.csv"))
        extension = "CSV"
    else:
        files = list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls"))
        extension = "Excel"
    
    if not files:
        print(f"\nNo {extension} files found in {directory}/")
        file_path = input(f"Enter full path to your {extension} file: ").strip()
        return Path(file_path)
    
    print(f"\nAvailable {extension} files in {directory}/:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file.name}")
    
    print(f"{len(files) + 1}. Enter custom path")
    
    while True:
        try:
            choice = int(input(f"\nSelect {extension} file (1-{len(files) + 1}): "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
            elif choice == len(files) + 1:
                file_path = input(f"Enter full path to your {extension} file: ").strip()
                return Path(file_path)
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

@click.command()
@click.option('--csv', help='Path to CSV training file (optional - will prompt if not provided)')
@click.option('--xlsx', help='Path to XLSX test file (optional - will prompt if not provided)')
@click.option('--output', default='reports/', help='Output directory for reports')
@click.option('--model-path', default='models/trained_model.cbm', help='Path to save/load model')
@click.option('--retrain', is_flag=True, help='Force retrain the model')
@click.option('--interactive', is_flag=True, default=True, help='Interactive file selection mode')
def main(csv, xlsx, output, model_path, retrain, interactive):
    """Compare CSV training data against XLSX test data using ML anomaly detection."""
    
    print("🔍 CSV-XLSX Anomaly Detector")
    print("=" * 40)
    
    try:
        # Interactive file selection if paths not provided
        if not csv and interactive:
            csv = select_file("csv")
        elif not csv:
            raise click.BadParameter("CSV file path is required")
            
        if not xlsx and interactive:
            xlsx = select_file("xlsx")
        elif not xlsx:
            raise click.BadParameter("XLSX file path is required")
        
        # Validate file paths
        csv_path = Path(csv)
        xlsx_path = Path(xlsx)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not xlsx_path.exists():
            raise FileNotFoundError(f"XLSX file not found: {xlsx_path}")
        
        print(f"\n📁 Training CSV: {csv_path}")
        print(f"📁 Test XLSX: {xlsx_path}")
        
        # Create output directories
        Path(output).mkdir(parents=True, exist_ok=True)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Phase 1: Load and process data
        print("\n🔄 Loading and processing data...")
        processor = DataProcessor()
        csv_data = processor.load_csv(csv_path)
        xlsx_data = processor.load_xlsx(xlsx_path)
        
        # Align columns
        aligned_data = processor.align_columns(csv_data, xlsx_data)
        
        # Phase 2: Train or load model
        trainer = ModelTrainer()
        if retrain or not Path(model_path).exists():
            print("🤖 Training new CatBoost model...")
            model = trainer.train(csv_data)
            trainer.save_model(model, model_path)
        else:
            print("📦 Loading existing model...")
            model = trainer.load_model(model_path)
        
        # Phase 3: Detect anomalies
        print("🔍 Detecting anomalies and differences...")
        detector = AnomalyDetector(model, csv_data)
        differences = detector.compare_data(aligned_data)
        
        # Phase 4: Generate reports
        print("📊 Generating reports...")
        reporter = ReportGenerator()
        report_paths = reporter.generate_report(differences, output, csv_path.stem, xlsx_path.stem)
        
        # Results summary
        print("\n" + "=" * 50)
        print("✅ Analysis Complete!")
        print(f"📈 Found {len(differences)} differences")
        print(f"📄 Reports saved to:")
        for path in report_paths:
            print(f"   • {path}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == '__main__':
    main()