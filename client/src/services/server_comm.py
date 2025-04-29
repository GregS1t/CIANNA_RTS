import time
import requests
import os

def poll_for_completion(server_url, job_id, interval=5):
    """
    Periodically check if the UWS job is completed.

    Args:
        server_url (str): Server base URL.
        job_id (str): Job ID to monitor.
        interval (int): Polling interval in seconds.

    Returns:
        bool: True if job completed, False if error or timeout.
    """
    while True:
        response = requests.get(f"{server_url}/jobs/{job_id}")
        if response.status_code == 200:
            phase = response.json().get("phase")
            if phase == "COMPLETED":
                print(f"[CLIENT] Job {job_id} completed.")
                return True
            elif phase == "ERROR":
                print(f"[CLIENT] Job {job_id} encountered an error.")
                return False
        time.sleep(interval)


def download_result(server_url, job_id, destination_folder):
    """
    Download result file from server after job completion.

    Args:
        server_url (str): Server base URL.
        job_id (str): Job ID to retrieve.
        destination_folder (str): Local folder to save the result.

    Returns:
        str: Path to the saved result file, or None if failed.
    """
    response = requests.get(f"{server_url}/jobs/{job_id}/results", stream=True)
    if response.status_code == 200:
        os.makedirs(destination_folder, exist_ok=True)
        file_path = os.path.join(destination_folder, f"net0_rts_{job_id}.dat")
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[CLIENT] Result saved at {file_path}")
        return file_path
    else:
        print("[CLIENT] Download error:", response.text)
        return None
