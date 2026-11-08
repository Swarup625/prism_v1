@echo off
title PRISM - Prediction Reliability & Self-Correction
echo =========================================================
echo  PRISM: Prediction Reliability and Self-Correction System
echo =========================================================
echo.
echo Starting PRISM Standalone Server...
echo Serving Web Application and REST API on http://localhost:8000
echo.
python backend\server.py 8000
pause

