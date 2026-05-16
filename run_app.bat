@echo off
echo ===================================================
echo   🏥 Medical Insurance Predictor - Setup & Run
echo ===================================================
echo.

echo [1/3] Installing required Python libraries...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo ❌ ERROR: Failed to install requirements. Please ensure Python and Pip are installed.
    pause
    exit /b
)

echo.
echo [2/3] Training the AI model on real-world data...
python train_model.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ ERROR: Failed to train the model.
    pause
    exit /b
)

echo.
echo [3/3] Launching the Web Application...
echo.
echo 💡 The app will open in your default browser shortly.
echo 💡 Press Ctrl+C in this window to stop the app.
echo.
streamlit run app.py

pause
