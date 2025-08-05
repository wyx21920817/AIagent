@echo off
REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Install requirements if needed
pip install -r requirements.txt

REM Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

REM Deactivate virtual environment
if exist "venv\Scripts\deactivate.bat" (
    call venv\Scripts\deactivate.bat
)
