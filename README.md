# CSV-XLSX Anomaly Detector

🔍 **Find differences between your CSV and XLSX files using machine learning**

This tool trains on your CSV data to learn normal patterns, then scans your XLSX file row-by-row to find any differences or anomalies.

## 🚀 Quick Start

### 1. Download the Code
```bash
git clone https://github.com/teleman1991/csv-xlsx-anomaly-detector.git
cd csv-xlsx-anomaly-detector
```

### 2. Set Up Conda Environment
```bash
# Create new environment
conda create -n anomaly-detector python=3.9
conda activate anomaly-detector

# Install packages
pip install -r requirements.txt
```

### 3. Add Your Files
Put your data files in the `data/` folder:
```bash
# Copy your files to the data folder
cp /path/to/your/training_data.csv data/
cp /path/to/your/test_data.xlsx data/
```

### 4. Run the Analysis
```bash
# Simple run - it will ask you to pick files
python src/main.py
```

That's it! The tool will:
- ✅ Let you select your CSV and XLSX files
- ✅ Train a model on your CSV patterns  
- ✅ Compare every row in your XLSX file
- ✅ Generate detailed reports of all differences found

## 📁 What You Get

After running, check the `reports/` folder for:

- **📊 Excel Report** - Summary stats + detailed differences
- **📋 CSV Report** - Raw data for further analysis  
- **🌐 HTML Dashboard** - Interactive visual report

## 🎯 Example Usage

### Interactive Mode (Recommended)
```bash
conda activate anomaly-detector
python src/main.py
```
Then follow the prompts to select your files.

### Command Line Mode
```bash
# Specify files directly
python src/main.py --csv data/training.csv --xlsx data/test.xlsx

# Force retrain the model
python src/main.py --csv data/training.csv --xlsx data/test.xlsx --retrain

# Custom output folder
python src/main.py --csv data/training.csv --xlsx data/test.xlsx --output my_reports/
```

## 🔍 What It Detects

The tool finds several types of differences:

1. **📍 Exact Mismatches** - Values that don't match your training data
2. **📈 Statistical Outliers** - Numbers outside normal ranges
3. **🤖 Pattern Anomalies** - ML-detected deviations from learned patterns
4. **❌ Data Quality Issues** - Missing values, wrong types, etc.

## 📊 Understanding Your Reports

### Excel Report Sheets:
- **Summary** - Overview of total differences found
- **Detailed_Differences** - Every difference with explanations
- **By_Column_Analysis** - Which columns have the most issues
- **High_Severity_Only** - Just the critical problems

### Key Columns in Reports:
- **row_index** - Which row in your XLSX has the issue
- **column** - Which column has the difference
- **test_value** - What value was found in your XLSX
- **expected_pattern** - What the model expected based on training
- **severity** - high/medium/low priority
- **anomaly_score** - How unusual this difference is (0-1)

## 🛠 Troubleshooting

### "No matching columns found"
- Make sure your CSV and XLSX have some column names in common
- Column names are case-sensitive

### "No differences found"
- Your data matches perfectly! 🎉
- Try using `--retrain` to rebuild the model

### Memory issues with large files
- The tool processes row-by-row, so it should handle large files
- If you have issues, try splitting your XLSX into smaller chunks

### File encoding errors
- The tool tries multiple encodings automatically
- If problems persist, save your CSV as UTF-8

## 🔄 Workflow Tips

1. **Start small** - Test with a subset of your data first
2. **Check column alignment** - Make sure CSV and XLSX have matching column names  
3. **Review high severity first** - Focus on the most critical differences
4. **Retrain when needed** - Use `--retrain` if your data patterns change

## 📋 Requirements

- Python 3.8 or higher
- About 500MB disk space for dependencies
- Works on Windows, Mac, and Linux

## 💡 Need Help?

The tool shows progress as it runs and will tell you:
- ✅ How many rows/columns were loaded
- ✅ Model training progress  
- ✅ How many differences were found
- ✅ Where your reports were saved

Each report includes detailed explanations of what each difference means and how critical it is.

---
**Ready to find those differences? Run `python src/main.py` and let's get started! 🚀**