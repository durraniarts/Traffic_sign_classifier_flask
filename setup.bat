@echo off

echo Setting up the virtual environment...
python -m venv .venv

echo Activating the virtual environment...
call .venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo Setup complete. Virtual environment activated.