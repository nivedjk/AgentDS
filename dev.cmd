@echo off
REM AgentDS dev launcher (Windows) - thin wrapper over run.py.
REM Usage: dev  [--backend-only ^| --frontend-only ^| --no-open]
python "%~dp0run.py" %*
