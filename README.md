# Police Officer-to-Case Matching ML System

A machine learning system that intelligently matches police officers to cases based on historical data, officer expertise, and case characteristics.

## Project Overview

This system uses machine learning to recommend the most suitable police officers for incoming cases by analyzing:
- **Officer profiles** (experience, skills, language abilities, success rates)
- **Case characteristics** (crime type, location, complexity, urgency)
- **Historical assignments** (which officers resolved similar cases successfully)

## Features

✅ Synthetic dataset generation with realistic patterns
✅ Comprehensive data preprocessing and exploration
✅ Advanced feature engineering for officers and cases
✅ Multiple ML model training (Logistic Regression, Random Forest, Gradient Boosting)
✅ Model comparison and automatic best-model selection
✅ Detailed evaluation with visualizations
✅ Officer ranking system for new cases
✅ Compatibility scoring between officers and cases

## Project Structure

```
police-case-matching-ml/
├── data/
│   ├── generate_data.py          # Synthetic data generation
│   ├── officers.csv              # Officers dataset
│   ├── cases.csv                 # Cases dataset
│   └── assignments_history.csv   # Historical assignments
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Data loading & preprocessing
│   ├── feature_engineering.py    # Feature creation & officer ranking
│   ├── model_training.py         # Model training & comparison
│   └── evaluation.py             # Model evaluation & visualization
├── models/
│   ├── best_model.pkl            # Trained model
│   ├── preprocessor.pkl          # Preprocessing objects
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── prediction_distribution.png
│   ├── feature_importance.png
│   └── evaluation_report.txt
├── notebooks/
│   └── analysis.ipynb            # Jupyter notebook for exploration
├── main.py                        # Main pipeline execution
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mdNafish1527/police-case-matching-ml.git
cd police-case-matching-ml
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete pipeline:

```bash
python main.py
```

This will:
1. ✅ Generate synthetic dataset (100 officers, 500 cases, 400 historical assignments)
2. ✅ Preprocess and explore the data
3. ✅ Engineer advanced features
4. ✅ Train 3 different ML models
5. ✅ Compare models and select the best
6. ✅ Generate evaluation plots and reports
7. ✅ Save trained model and preprocessor
8. ✅ Demonstrate officer ranking for a sample case

### Output

After running `main.py`, you'll find:

- **Data Files**: `data/officers.csv`, `data/cases.csv`, `data/assignments_history.csv`
- **Models**: `models/best_model.pkl`, `models/preprocessor.pkl`
- **Visualizations**: 
  - Confusion Matrix
  - ROC-AUC Curve
  - Prediction Probability Distribution
  - Feature Importance Chart
- **Report**: `models/evaluation_report.txt` with detailed metrics

## Dataset Structure

### Officers Dataset

| Field | Type | Description |
|-------|------|-------------|
| officer_id | string | Unique officer identifier |
| years_experience | int | Years on the force |
| special_units | string | Specialized training (pipe-separated) |
| languages | string | Languages spoken (pipe-separated) |
| region | string | Assigned region |
| solved_cases | int | Number of successfully resolved cases |
| success_rate | float | Percentage of cases resolved |
| availability | bool | Currently available for assignment |
| cases_assigned | int | Current active cases |

### Cases Dataset

| Field | Type | Description |
|-------|------|-------------|
| case_id | string | Unique case identifier |
| crime_type | string | Type of crime |
| location | string | Case location |
| complexity | string | Case complexity level |
| complexity_score | int | Numerical complexity (1-10) |
| language_required | string | Language needed |
| urgency | string | Urgency level |
| case_date | string | Date case filed |
| description_length | int | Case description length |


## Machine Learning Models

### Models Trained

1. **Logistic Regression**
   - Fast, interpretable baseline model
   - Good for initial understanding of features

2. **Random Forest**
   - Ensemble of decision trees
   - Handles non-linear relationships
   - Feature importance analysis

3. **Gradient Boosting**
   - Sequential tree building
   - Often achieves best performance
   - Captures complex patterns

### Model Selection

The system automatically selects the best model based on **F1-Score**, which balances precision and recall for binary classification.

## Key Features

### Feature Engineering

**Officer Features:**
- Experience level categories (Junior, Mid-Level, Senior, Expert)
- Success rate categories
- Workload score
- Officer strength score
- Count of special units and languages
- Availability score

**Case Features:**
- Urgency score
- Case difficulty score
- Description complexity indicator

**Compatibility Scoring:**
- Experience match
- Location match
- Language match
- Officer success rate
- Availability
- Workload consideration

### Officer Ranking Algorithm

For each new case, the system:
1. Calculates compatibility score for all officers
2. Factors in experience, location, language, success rate
3. Considers current workload and availability
4. Returns top-N recommended officers with scores

## Performance Metrics

The system evaluates models using:

- **Accuracy**: Overall correctness
- **Precision**: When we predict resolution, how often correct
- **Recall**: Percentage of actual resolutions identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## Example Output

```
Top 5 Recommended Officers for Robbery Case:

  officer_id  years_experience  success_rate  region  availability  compatibility_score
  OFF0045     18                0.92          North   True          94.2
  OFF0023     15                0.88          North   True          91.5
  OFF0078     20                0.90          Downtown False         88.7
  OFF0031     12                0.85          North   True          85.3
  OFF0056     14                0.82          North   True          82.9
```

## Customization

### Adjust Training Parameters

Edit `main.py`:
- Number of officers: `generate_officers(100)`
- Number of cases: `generate_cases(500)`
- Historical assignments: `generate_assignments(..., 400)`
- Test size: `trainer.split_data(..., test_size=0.2)`

### Modify Feature Engineering

Edit `src/feature_engineering.py` to:
- Add new officer features
- Add new case features
- Adjust compatibility scoring weights

## Future Improvements

- [ ] Integration with real police department data
- [ ] Real-time case assignment API
- [ ] Web dashboard for dispatchers
- [ ] Feedback loop for continuous model improvement
- [ ] NLP for case description analysis
- [ ] Geographic optimization
- [ ] Time-series analysis for trend detection
- [ ] Multi-officer team recommendations

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning
- matplotlib: Data visualization
- seaborn: Statistical visualization
- joblib: Model persistence

## License

IIT- University Of Dhaka

## Author

Created by: mdNafish1527

## Contributing

Contributions welcome! Please open issues and pull requests.

## Support

For questions or issues, please open a GitHub issue.
