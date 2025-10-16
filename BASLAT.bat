@echo off
chcp 65001 >nul
color 0A

echo.
echo YUZ VE EL TAKIP SISTEMI
echo.

if not exist "models\deploy.prototxt" (
    echo Modeller indiriliyor...
    py modelleri_indir.py
    echo.
)

py yuz_el_takip.py

pause
