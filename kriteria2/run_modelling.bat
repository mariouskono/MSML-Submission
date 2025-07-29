@echo off
setlocal

echo --------------------------------------
echo [1] Setup environment
echo --------------------------------------
if not exist "venv\" (
  python -m venv venv
)
call venv\Scripts\activate

pip install --upgrade pip setuptools wheel

if exist requirements.txt (
  pip install -r requirements.txt
)

echo --------------------------------------
echo [2] Run modelling scripts
echo --------------------------------------
python modelling.py || exit /b 1
python modelling_tuning.py || exit /b 1

echo --------------------------------------
echo [3] Start MLflow UI
echo --------------------------------------
set MLFLOW_PORT=5000
netstat -ano | findstr :%MLFLOW_PORT% >nul
if %errorlevel%==0 (
  echo Port %MLFLOW_PORT% is occupied. Switching to 5001.
  set MLFLOW_PORT=5001
)

start cmd /k "call venv\Scripts\activate && mlflow ui --port %MLFLOW_PORT%"
timeout /t 5 >nul
start "" http://localhost:%MLFLOW_PORT%

endlocal