import os
import json
from threading import Lock
from datetime import datetime

# Constants
JOB_LOG_PATH = os.path.join(os.getcwd(), 'data', 'job_log.json')
BACKUP_FOLDER = os.path.join(os.getcwd(), 'data', 'log_backups')
os.makedirs(BACKUP_FOLDER, exist_ok=True)

# Lock for concurrent access
log_lock = Lock()

# Global priority counter (can be loaded/saved for persistence)
Job_priority = 0

def load_job_log():
    """Load the job log from the JSON file."""
    if not os.path.exists(JOB_LOG_PATH):
        return {}
    with open(JOB_LOG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_job_log(log_data):
    """Save the job log and create a timestamped backup."""
    with open(JOB_LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)

    # Create backup with timestamp
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(BACKUP_FOLDER, f'job_log_{timestamp}.json')
    with open(backup_path, 'w', encoding='utf-8') as bf:
        json.dump(log_data, bf, indent=2)

def log_new_job(process_id, fits_path, reception_date):
    """Add a new job entry to the log."""
    global Job_priority
    with log_lock:
        log_data = load_job_log()
        Job_priority += 1

        log_data[process_id] = {
            "jobId": process_id,
            "imagePath": fits_path,
            "model": "",
            "quantization": "",
            "priority": Job_priority,
            "status": "PENDING",
            "phase": "PENDING",
            "receptionDate": reception_date,
            "end_time": "",
            "comment": ""
        }

        save_job_log(log_data)

def update_job_status(process_id, **kwargs):
    """Update fields of an existing job entry."""
    with log_lock:
        log_data = load_job_log()
        job = log_data.get(process_id)
        if not job:
            print(f"[job_logger] Warning: job {process_id} not found in log.")
            return

        for key, value in kwargs.items():
            if value is not None:
                job[key] = value

        save_job_log(log_data)