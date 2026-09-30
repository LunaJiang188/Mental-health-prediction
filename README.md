# Mental Health Prediction

## Project Overview

This project analyses a Kaggle dataset containing demographic, lifestyle, socioeconomic, and health-related information.
The project has two main objectives:

1. Explore the dataset and identify patterns using EDA and additional statistical methods.
2. Develop a ML model to predict whether an individual has a **reported history of mental illness**.

The target variable used for prediction is: `History_of_Mental_Illness`

------

## Dataset

The dataset used for this project is the Kaggle Depression Dataset: https://www.kaggle.com/datasets/anthonytherrien/depression-dataset/data

------

## Project Structure

```text
Mental-health-prediction/
│
├── README.md
├── requirements.txt
|__ .gitignore
|
│
├── data/
│   └── mental_health_data.csv
|
|__models/
|   |── preprocessor.pkl
|   └── mental_health_xgboost.pkl
│
├── src/
│   ├── data_explore.ipynb
│   |── modelling.ipynb
│   |── mini_signoff.ipynb
│   └── mini_signoff.html
│
|── src/
│   ├── __init__.py
│   └── interpretation.py
|
├── test/
|    └── interpretation.py
|
├── figures/
|    └── ....
|
``` 
------
## How to start
### 1. Clone the Repository

Clone the Git repository to your local machine:

```bash
git clone https://github.com/LunaJiang188/mental-health-prediction.git
cd mental-health-prediction
```

### 2. Create a Virtual Environment

If you have Windows system

```bash
python -m venv .venv
.venv\Scripts\activate
```

If you have macOS / Linux system

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, you should see `(.venv)` in your terminal.

### 3. Install the Required Libraries

Install the required Python packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter Notebook

The analysis and modelling work is provided in the `src` folder.

Start Jupyter with:

```bash
jupyter notebook
```

Then open:

```text
src/data_explore.ipynb
```

or:

```text
src/modelling.ipynb
```

### 5. Download the Dataset
The dataset is downloaded from Kaggle using `kagglehub`.

After the initial data exploration, I manually added `RecordID` and `data_split` columns. To keep the data consistent and ensure that the test data remains unseen until the final evaluation, I saved the updated dataset as `data/mental_health_data.csv`.