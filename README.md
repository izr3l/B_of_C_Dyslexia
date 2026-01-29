# Dyslexia Detection AI

## Overview
This project is a Machine Learning application designed for the early detection of dyslexia using behavioral and reading metrics. It utilizes various classification algorithms to predict the likelihood of dyslexia based on features such as reading speed, accuracy, and reaction times. The application is built with **Streamlit** to provide an interactive and user-friendly interface for exploratory data analysis (EDA), model training, and live predictions.

## Features
- **Project Info**: Overview of the project goals, approach, and team members.
- **Exploratory Data Analysis (EDA)**: Interactive visualizations to understand data distributions, correlations, and target variable balance.
- **Model Playground**: Interactive environment to train and test **Random Forest** and **SVM** models with adjustable hyperparameters.
- **Model Comparison**: Automatically train and compare the selected models (Random Forest vs. SVM) to identify the best performer.
- **Live Prediction**: Input new behavioral data points to get a real-time risk assessment (Dyslexic vs. Non-Dyslexic). The system automatically handles features like total clicks.

## Tech Stack
- **Language**: Python
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy, **Imbalanced-learn (SMOTE)**
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn

## Installation

1.  **Clone the repository** (if applicable) or download the project files.
2.  **Navigate to the project directory**:
    ```bash
    cd "path/to/Dyslexia"
    ```
3.  **Create a virtual environment** (Recommended):
    - Windows:
      ```bash
      python -m venv env
      .\env\Scripts\activate
      ```
    - Mac/Linux:
      ```bash
      python3 -m venv env
      source env/bin/activate
      ```
4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Ensure the dataset `dyslexia_synthetic_4237.csv` is present in the root directory.
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
    *Alternatively, on Windows, you can simply run the provided `run_app.bat` script.*
3.  The application will open in your default web browser (usually at `http://localhost:8501`).

## Project Structure
```
Dyslexia/
├── app.py                  # Main Streamlit application entry point
├── dyslexia_synthetic_4237.csv # Dataset file (required)
├── requirements.txt        # Python dependencies
├── run_app.bat             # Windows batch script to run the app
├── README.md               # Project documentation
├── src/                    # Source code modules
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── visualization.py
└── .vscode/                # VS Code configuration
```

## Team Members
- Farhan Ahmad Nasiruddeen
- Israel Oluwabukunmi Olayemi
- Mahmud Yusuf Aminu
- Makinde Mark Olusanya
- Abdulmuhaimin Muhammad

## Dataset
The model is trained on data containing behavioral metrics such as:
- Vocabulary scores
- Short-term memory task performance
- Reaction times
- Demographic info (Age, Gender)

*Note: This tool is intended for screening and educational purposes and should not replace professional medical diagnosis.*
