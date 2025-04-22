import os
import random
import json
from src.core.xml_utils import create_xml_param
from src.core.file_transfer import send_xml_fits_to_server
from src.services.server_comm import poll_for_completion, download_result
from src.utils.cianna_xml_updater import update_cianna_models
from src.utils.ssh_tunnel import create_ssh_tunnel


DESTINATION_FOLDER = "results"


def load_config(config_path):
    """
    Load and parse a JSON configuration file robustly.

    Parameters:
        config_path (str): The file path to the configuration JSON file.

    Returns:
        dict: A dictionary containing the configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from file {config_path}: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the config file {config_path}: {e}")
    
    return config

def emulate_client_request(server_url, image_path, request_number, config):
    user_id = 2443423 + request_number
    ra = random.uniform(0, 360)
    dec = random.uniform(-90, 90)
    h = random.randint(50, 200)
    w = random.randint(50, 200)
    yolo_model = config.get("YOLO_MODEL", "net0_s1800.dat")
    quantization = config.get("QUANTIZATION", "FP32C_FP32A")
    
    xml_data = create_xml_param(user_id, ra, dec, h, w, image_path, yolo_model, quantization)
    process_id = send_xml_fits_to_server(server_url, xml_data)
    if process_id is None:
        print(f"Error for request {request_number}")
    else:
        print(f"Request {request_number} sent successfully with process ID: {process_id}")
        # if poll_for_completion(server_url, process_id):
        #     download_result(server_url, process_id,
        #                     destination_folder=DESTINATION_FOLDER)
        # else:
        #     print(f"Error for request : {request_number}")

def main():
    """
    Main function for the client.

    Steps:
      1. Load the client configuration.
      2. If a remote connection is specified, establish an SSH tunnel.
      3. Update the local CIAnna models XML file by always retrieving the latest version.
      4. Locate FITS images from the designated input folder.
      5. Emulate a set number of client requests to the server.
    """
    # Load configuration from JSON file
    config = load_config("configs/param_cianna_rts_client.json")
    
    # Path to the local Cianna models XML file
    local_models_file = config.get("LOCAL_FILE_MODELS")

    # Determine connection mode and set server URL accordingly.
    print(40 * "-.")
    client_connexion = config.get("CLIENT_CONNEXION", "local").lower()
    tunnel = None
    if client_connexion == "remote":
        print("Establishing remote connection via SSH tunnel...")
        tunnel = create_ssh_tunnel(
            ssh_server_ip = config.get("SSH_SERVER_IP"),
            ssh_username  = config.get("SSH_USERNAME"),
            ssh_password  = config.get("SSH_PASSWORD"),
            remote_port   = int(config.get("REMOTE_PORT", 3000)),
            local_port    = int(config.get("LOCAL_PORT", 3000))
        )
        server_url = f"http://127.0.0.1:{tunnel.local_bind_port}"
    else:

        server_url = f"http://127.0.0.1:{config.get('LOCAL_PORT', 3000)}"
    
    print(f"Connecting to a {client_connexion} server...")
    print("Server URL:", server_url)
    print(40 * "-.")
    # Update the local Cianna models XML file (always retrieves the latest version)
    models_url = f"{server_url}/models/CIANNA_models.xml"
    update_result = update_cianna_models(models_url, local_models_file)
    if update_result is None:
        print("Error updating CIANNA models.")
        if tunnel is not None:
            tunnel.stop()
        return

    # Get list if images for test
    image_folder = "../../DIR_images" 
    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".fits")]
    if not images:
        print("No fits images in ", image_folder)
        if tunnel is not None:
            tunnel.stop()
        return
    
    # Emuler 3 requêtes client
    nb_requests = 1
    for i in range(nb_requests):
        image_path = random.choice(images)
        print(f"\n\nRequest #{i+1} with image {image_path}")
        emulate_client_request(server_url, image_path, i+1, config)
        print(40 * "-.")

    # If an SSH tunnel was established, stop it after finishing
    if tunnel is not None:
        tunnel.stop()
        print("SSH tunnel closed.")

if __name__ == '__main__':
    main()

