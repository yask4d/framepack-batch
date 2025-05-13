@echo off

call environment.bat

cd %~dp0webui

"%DIR%\python\python.exe" batch.py --server 127.0.0.1 --inbrowser

:done
pause