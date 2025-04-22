import time
import requests
import os

def poll_for_completion(server_url, process_id, interval=5):
    """
    Check periodically if the process is completed
    """
    while True:
        response = requests.get(f"{server_url}/status/{process_id}")
        if response.status_code == 200:
            status = response.json().get("status")
            if status == "COMPLETED":
                print(f"Process {process_id} ended.")
                return True
            elif status == "ERROR":
                print(f"Error in process {process_id}.")
                return False
        time.sleep(interval)

def download_result(server_url, process_id, destination_folder):
    """
    Download file from server
    """
    response = requests.get(f"{server_url}/download/{process_id}")
    if response.status_code == 200:
        # Vérifier et créer le dossier destination s'il n'existe pas
        os.makedirs(destination_folder, exist_ok=True)
        
        file_path = os.path.join(destination_folder, f"result_{process_id}.dat")
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Result saved here: {file_path}")
        return file_path
    else:
        print("Download error:", response.text)
        return None

