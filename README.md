# Sentiment Analysis on Movie Reviews

A machine learning project for binary sentiment classification (`POSITIVE`/`NEGATIVE`) from movie review text.

## Overview

This repository contains an end-to-end NLP workflow implemented in `Sentiment_Analysis.ipynb`, covering:
- Exploratory data analysis (EDA)
- Text preprocessing and cleaning
- TF-IDF vectorization
- Feature selection and normalization
- Model training and evaluation
- Hyperparameter tuning experiments
- Test-set prediction and submission file generation

## Project Structure

```text
Sentiment_Analysis/
|-- Sentiment_Analysis.ipynb
|-- README.md
`-- Data/
    |-- train.csv
    |-- test.csv
    |-- movies.csv
    `-- sample.csv
```

## Dataset

Primary files used by the notebook:
- `Data/train.csv` (~162,848 rows): training data with sentiment labels
- `Data/test.csv` (~55,351 rows): unlabeled review data for prediction
- `Data/movies.csv` (~143,258 rows): movie metadata (present but mostly unused in modeling)
- `Data/sample.csv` (~55,315 rows): sample submission format (`id`, `sentiment`)

Main fields:
- Training: `reviewText`, `sentiment` (+ metadata columns)
- Testing: `reviewText` (+ metadata columns)

## Methods and Pipeline

The notebook follows this modeling pipeline:
1. Load and inspect data
2. Handle missing values (`SimpleImputer`)
3. Encode target labels (`LabelEncoder`)
4. Clean text with regex (remove URLs, mentions/hashtags, numbers, bracketed text, tabs/newlines)
5. Convert text to TF-IDF features (`TfidfVectorizer`)
6. Reduce and normalize features (`SelectKBest(chi2, k=40000)` + `Normalizer`)
7. Train/test split (`test_size=0.3`, `random_state=42`)
8. Train and evaluate baseline models:
   - `LinearSVC`
   - `LogisticRegression`
9. Run GridSearchCV experiments on:
   - `LogisticRegression`
   - `MultinomialNB`
   - `RandomForestClassifier`
   - `SVC`
10. Generate final predictions and export CSV

## Tech Stack

- Python 3.9+
- Jupyter Notebook
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Setup

1. Clone the repository and open the project root.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

4. Start Jupyter:

```bash
jupyter notebook
```

5. Open `Sentiment_Analysis.ipynb` and run cells sequentially.

## Important Note on File Paths

The notebook currently reads CSV files using absolute local paths in some cells.
For portability, replace those with project-relative paths, for example:

```python
Trainset = pd.read_csv("Data/train.csv")
Testset = pd.read_csv("Data/test.csv")
Movies = pd.read_csv("Data/movies.csv")
```

## Output

The notebook writes predictions to:
- `testOutput.csv`

Output schema:
- `id`
- `sentiment` (`POSITIVE` or `NEGATIVE`)

## Evaluation

Model quality is assessed using:
- `classification_report`
- confusion matrix
- accuracy score

The notebook observations indicate class imbalance toward positive sentiment, which may influence model behavior.

## Future Improvements

- Add a `requirements.txt` for deterministic dependency management
- Move preprocessing and modeling into reusable Python modules/scripts
- Add cross-validation and stronger imbalance handling
- Track experiments and metrics systematically
- Add unit tests for preprocessing utilities
