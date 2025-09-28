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

### 3. Prepare Your Data Files

You have **two options** for organizing your files:

#### Option A: Use the data folder (recommended)
```bash
# Create the data folder
mkdir data

# Copy your files there
cp /path/to/your/training_data.csv data/
cp /path/to/your/test_data.xlsx data/
```

#### Option B: Keep files anywhere
Keep your CSV and XLSX files wherever they are - you can specify full paths when running the tool.

### 4. Run the Analysis
```bash
# Interactive mode - will ask you to select files
python src/main.py
```

**If using Option A (data folder):**
- The tool will automatically find files in the `data/` folder
- Just select from the numbered list

**If using Option B (files anywhere):**
- Choose "Enter custom path" when prompted
- Type the full path like: `/Users/yourname/Documents/my_data.csv`

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
The tool will show you available files and let you pick them.

### Command Line Mode
```bash
# If files are in data folder
python src/main.py --csv data/training.csv --xlsx data/test.xlsx

# If files are anywhere else
python src/main.py --csv /full/path/to/training.csv --xlsx /full/path/to/test.xlsx

# Force retrain the model
python src/main.py --csv data/training.csv --xlsx data/test.xlsx --retrain

# Custom output folder
python src/main.py --csv data/training.csv --xlsx data/test.xlsx --output my_reports/
```

## 📂 Project Structure
```
csv-xlsx-anomaly-detector/
├── src/                    # Python code files
├── data/                   # Put your CSV/XLSX files here (optional)
├── models/                 # Trained models saved here
├── reports/                # Generated reports appear here
├── requirements.txt        # Dependencies
└── README.md              # This file
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

### "No CSV/XLSX files found in data/"
- **Solution 1:** Put your files in the `data/` folder
- **Solution 2:** Use custom paths when prompted or use command line mode

### "No matching columns found"
- Make sure your CSV and XLSX have some column names in common
- Column names are case-sensitive
- Check that both files have headers in the first row

### "File not found"
- Double-check your file paths
- Make sure file extensions are correct (.csv, .xlsx, .xls)
- Use full paths if files are outside the project folder

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
2. **Use the data folder** - Keeps everything organized
3. **Check column alignment** - Make sure CSV and XLSX have matching column names  
4. **Review high severity first** - Focus on the most critical differences
5. **Retrain when needed** - Use `--retrain` if your data patterns change

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