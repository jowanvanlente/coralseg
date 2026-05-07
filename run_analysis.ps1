# Launches the annotation analysis dashboard on port 8502
# (separate from the main webapp so both can run side by side)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"
streamlit run "$PSScriptRoot\analysis_app.py" --server.port 8502
