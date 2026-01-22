@echo off
if not exist "env" (
    echo Creating virtual environment...
    python -m venv env
)

echo Activating virtual environment...
call env\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Running Streamlit App...
streamlit run app.py
