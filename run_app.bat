@echo off
REM Lanza la app Streamlit con el interprete Python 3.10 correcto (donde estan torch/streamlit/monai).
REM Usa el puerto 8400 porque Windows tiene excluido el rango 8501-8907 (reservado por Hyper-V/WSL/Docker),
REM que es el rango por defecto de Streamlit y causa el error WinError 10013.
REM Doble-click sobre este archivo, o ejecutalo desde PowerShell: .\run_app.bat
cd /d "%~dp0"
"C:\Users\Sebas\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run app/streamlit_app.py --server.port 8400 --server.address localhost
pause
