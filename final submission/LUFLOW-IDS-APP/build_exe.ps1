param(
    [string]$Python = "python",
    [string]$Name = "LUFLOW-IDS"
)

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name $Name `
    --icon icon.ico `
    --hidden-import PyQt5.sip `
    --hidden-import xgboost `
    --hidden-import pyshark `
    --collect-submodules PyQt5 `
    --additional-hooks-dir hooks `
    --add-data "xgboost_models;xgboost_models" `
    --add-data "network-intrusion-xgboost.ipynb;." `
    --add-data "icon.ico;." `
    main.py
