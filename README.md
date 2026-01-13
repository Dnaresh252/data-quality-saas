# Data Quality Analysis Platform

A production-ready web application for automated data quality analysis and cleaning. Built to help data scientists and analysts quickly identify and fix data quality issues in CSV datasets.

## Features

- **Comprehensive Data Profiling**: Get detailed statistics about your dataset including shape, memory usage, and column types
- **Missing Value Detection**: Identify missing data patterns with intelligent imputation recommendations
- **Duplicate Analysis**: Find exact and subset duplicates with smart grouping
- **Outlier Detection**: Uses IQR method, Z-score, and Isolation Forest for robust outlier identification
- **Data Type Validation**: Automatically detect incorrect data types and get conversion suggestions
- **Inconsistency Detection**: Find formatting issues, case inconsistencies, and special characters
- **Correlation Analysis**: Identify highly correlated features and potential multicollinearity
- **Quality Scoring**: Overall data quality score from 0-100 with detailed breakdown
- **Automated Cleaning**: Download cleaned datasets with configurable cleaning strategies

## Tech Stack

### Backend
- FastAPI for REST API
- Pandas for data manipulation
- NumPy for numerical operations
- Scikit-learn for machine learning algorithms
- SciPy for statistical analysis

### Frontend
- Next.js 14 with App Router
- TypeScript for type safety
- Tailwind CSS for styling
- Axios for API calls
- Recharts for data visualization

## Installation

### Prerequisites
- Python 3.8 or higher
- Node.js 18 or higher
- npm or yarn

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

## Configuration

Create a `.env.local` file in the frontend directory:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Running the Application

### Start Backend Server
```bash
cd backend
python -m uvicorn api:app --reload
```

Backend will run on http://localhost:8000

### Start Frontend Server
```bash
cd frontend
npm run dev
```

Frontend will run on http://localhost:3000

## Usage

1. Navigate to http://localhost:3000
2. Click "Start Analysis" or go to the Analyze page
3. Upload your CSV file (max 100MB)
4. View comprehensive data quality report
5. Download cleaned dataset

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /analyze` - Analyze dataset and get quality report
- `POST /clean` - Clean dataset with custom strategies
- `GET /download/{job_id}` - Download cleaned CSV file
- `DELETE /job/{job_id}` - Delete job data
- `GET /jobs` - List all active jobs

## Project Structure
```
data-quality-saas/
├── backend/
│   ├── modules/
│   │   ├── run_pipeline.py       # Main analysis pipeline
│   │   ├── profiling.py          # Dataset profiling
│   │   ├── correlations.py       # Correlation analysis
│   │   ├── missing_values.py     # Missing data detection
│   │   ├── duplicates.py         # Duplicate detection
│   │   ├── outliers.py           # Outlier detection
│   │   ├── inconsistencies.py    # Inconsistency detection
│   │   ├── data_types.py         # Data type validation
│   │   ├── drift.py              # Data drift detection
│   │   └── cleaning.py           # Data cleaning functions
│   ├── api.py                    # FastAPI application
│   └── requirements.txt          # Python dependencies
├── frontend/
│   ├── app/                      # Next.js app directory
│   ├── components/               # React components
│   ├── lib/                      # Utility functions
│   └── types/                    # TypeScript types
└── README.md
```

## Analysis Methods

### Missing Values
- Counts missing values per column
- Calculates missing percentage
- Identifies missing patterns
- Provides imputation strategies

### Duplicates
- Detects exact duplicates
- Finds subset duplicates (ignoring ID columns)
- Groups duplicate rows
- Calculates severity levels

### Outliers
- IQR (Interquartile Range) method
- Z-score analysis
- Isolation Forest algorithm
- Per-column outlier statistics

### Data Types
- Mixed type detection
- Date format identification
- Boolean column detection
- High cardinality string detection

### Correlations
- Pearson correlation for numeric features
- Chi-square test for categorical associations
- Cramér's V for association strength
- Multicollinearity warnings

## Cleaning Strategies

### Duplicate Handling
- `keep`: Keep all duplicates
- `flag`: Add is_duplicate column
- `remove`: Remove duplicates
- `auto`: Remove if severity is high

### Missing Value Imputation
- `median`: Fill with median (numeric)
- `mean`: Fill with mean (numeric)
- `mode`: Fill with mode (categorical)
- `zero`: Fill with zero

### Outlier Treatment
- `clip`: Clip to IQR bounds
- `flag`: Add outlier flag column
- `remove`: Remove outlier rows

## Performance

- Handles datasets up to 100MB
- Automatic encoding detection (UTF-8, Latin1, etc.)
- Smart delimiter detection (comma, semicolon, tab, pipe)
- Memory-efficient processing
- Streaming file downloads

## Future Improvements

- Add support for Excel files
- Implement batch processing for multiple files
- Add data visualization charts
- Export reports as PDF
- Add user authentication
- Implement file history tracking

## Contributing

This is a student project built for learning purposes. Feedback and suggestions are welcome.

## License

MIT License

## Contact

For questions or feedback, please open an issue on GitHub.
