#!/bin/bash

SERVICE_NAME="job_scheduler"
USER_NAME=$(whoami)
PROJECT_DIR=$(pwd)
PYTHON_PATH=$(which python3)
SCRIPT_PATH="$PROJECT_DIR/src/processing/job_scheduler.py"

# Fichier systemd temporaire
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "[INFO] Installing service: $SERVICE_NAME"
echo "[INFO] Using Python at: $PYTHON_PATH"
echo "[INFO] Project path: $PROJECT_DIR"

# Vérification que le fichier job_scheduler.py existe
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "[ERROR] Cannot find job_scheduler.py in $PROJECT_DIR"
  exit 1
fi

# Création du fichier de service
sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=CIANNA RTS Job Scheduler
After=network.target

[Service]
Type=simple
ExecStart=$PYTHON_PATH $SCRIPT_PATH
WorkingDirectory=$PROJECT_DIR
Restart=always
RestartSec=5
User=$USER_NAME
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# Rechargement systemd et activation
echo "[INFO] Reloading systemd and enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl start "$SERVICE_NAME"

echo "[SUCCESS] Service '$SERVICE_NAME' is now running."
sudo systemctl status "$SERVICE_NAME" --no-pager

