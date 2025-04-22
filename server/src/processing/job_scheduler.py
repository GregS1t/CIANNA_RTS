import os
import shutil
from src.utils.job_logger import load_job_log, update_job_status
from src.processing.pipeline import run_prediction_job

WAITING_DIR = "JOBS/WAITING"
ON_GOING_DIR = "JOBS/ON_GOING"
COMPLETED_DIR = "JOBS/COMPLETED"

def find_next_job_to_run():
    """
    Select the job with the lowest priority in WAITING.
    
    Returns:
        str or None: The process_id to run next, or None if none found.
    """
    log = load_job_log()
    waiting_jobs = {
        pid: entry for pid, entry in log.items()
        if entry["status"] == "WAITING" and os.path.isdir(os.path.join(WAITING_DIR, pid))
    }
    if not waiting_jobs:
        return None

    # Select job with lowest priority
    next_job = min(waiting_jobs.items(), key=lambda item: item[1]["priority"])
    return next_job[0]

def move_job_to_on_going(process_id):
    """
    Move a job from WAITING to ON_GOING.
    
    Args:
        process_id (str): Job identifier.
    
    Returns:
        str: Path to the new job folder in ON_GOING.
    """
    src = os.path.join(WAITING_DIR, process_id)
    dst = os.path.join(ON_GOING_DIR, process_id)
    shutil.move(src, dst)
    return dst

def run_next_job():
    """
    Execute the next available job by priority.
    """
    process_id = find_next_job_to_run()
    if not process_id:
        print("No job to run.")
        return

    print(f"[SCHEDULER] Selected job {process_id} for execution.")

    # Move job folder
    job_dir = move_job_to_on_going(process_id)
    update_job_status(process_id, status="ONGOING", comment="Moved to ON_GOING")

    # Paths to XML and FITS
    files = os.listdir(job_dir)
    xml_file = next((f for f in files if f.lower().endswith(".xml")), None)
    fits_file = next((f for f in files if f.lower().endswith(".fits")), None)

    if not xml_file or not fits_file:
        update_job_status(process_id, status="ERROR", comment="Missing XML or FITS in job folder")
        return

    xml_path = os.path.join(job_dir, xml_file)
    fits_path = os.path.join(job_dir, fits_file)

    run_prediction_job(process_id, xml_path, fits_path, ON_GOING_DIR)

if __name__ == "__main__":
    run_next_job()

