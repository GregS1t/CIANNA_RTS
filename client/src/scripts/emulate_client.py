import os, sys
import random
import json

import requests

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             "..", "..")))
from src.core.xml_utils import create_xml_param
from src.core.file_transfer import send_xml_fits_to_server
from src.services.server_comm import poll_for_completion, download_result
from src.utils.cianna_xml_updater import update_cianna_models, get_model_info
from src.utils.ssh_tunnel import create_ssh_tunnel
from src.utils.fits_utils import get_image_dim


DESTINATION_FOLDER = "results"
CONFIGS_DIR = os.path.join(os.getcwd(), 'client','configs')

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
    """
    Simulates a client sending a request to a server with astronomical image data.

    This function generates randomized parameters for a simulated observation 
    (RA, DEC, bounding box dimensions), creates an XML-formatted request including 
    user ID and model configuration, and sends it to the specified server. It logs 
    the result of the request, including success or failure status.

    Parameters:
    ----------
    server_url : str
        The base URL of the server to which the request will be sent.
    image_path : str
        The path to the image file (typically a FITS file) to be sent.
    request_number : int
        A unique number used to differentiate this request and compute the user ID.
    config : dict
        A dictionary of configuration options, including:
            - "YOLO_MODEL" (str, optional): Name of the YOLO model file to use.
            - "QUANTIZATION" (str, optional): Type of quantization for the model.
    """

    user_id = 2443423 + request_number
    ra  = random.uniform(0, 360)                            # Either the full image or a sub-image
    dec = random.uniform(-90, 90)
    h   = random.randint(50, 200)
    w   = random.randint(50, 200)
    yolo_model = "SDC1_Cornu_2024"  # Suppose to be selected by the user # Suppose to be selected by the user
    quantization = "FP32C_FP32A"    # Suppose to be selected by the user


    # Check if the image is compatible with the YOLO model
    model_info = get_model_info(config.get("LOCAL_FILE_MODELS"), yolo_model)

    print(f"[emulate_client_request] Model info: {model_info}")


    # 
    if model_info is None:
        print(f"Error: Model {yolo_model} not found in the local XML file.")
        return

    else:
        # Get the image dimensions
        image_info = get_image_dim(image_path)
        if image_info is None:
            print(f"Error: Unable to read image dimensions from {image_path}.")
            return
        image_size = image_info.get('shape', (0, 0))
        if image_size[0] < h or image_size[1] < w:
            print(f"Error: Image dimensions {image_size} are smaller than the requested bounding box ({h}, {w}).")
            return


        xml_data = create_xml_param(user_id, ra, dec, h, w, image_path, 
                                    yolo_model, quantization, model_info.get("Name"))

        process_id = send_xml_fits_to_server(server_url, xml_data)
        if process_id is None:
            print(f"[EMULATE] Error sending request {request_number}")
        else:
            print(f"[EMULATE] Request {request_number} sent successfully with process ID: {process_id}")

            try:
                # Poll for job completion
                print(f"[EMULATE] Polling for job {process_id} completion...")
                if poll_for_completion(server_url, process_id):
                    print(f"[EMULATE] Job {process_id} completed successfully.")
                    print(f"[EMULATE] Downloading result for job {process_id}...")
                    download_result(server_url, process_id, destination_folder=DESTINATION_FOLDER)
                    print(f"[EMULATE] Result for request {request_number} downloaded successfully.")
                else:
                    print(f"[EMULATE] Error: Job {process_id} did not complete successfully.")

            except requests.ConnectionError as e:
                print(f"[EMULATE] Network error while polling/downloading: {e}")
            except requests.Timeout as e:
                print(f"[EMULATE] Timeout error: {e}")
            except Exception as e:
                print(f"[EMULATE] Unexpected error: {e}")

def main():
    """
    Main function for the client.

    Steps:
      - Load the client configuration.
      - If a remote connection is specified, establish an SSH tunnel.
      - Update the local CIAnna models XML file by always retrieving the latest version.
      - Locate FITS images from the designated input folder.
      - Emulate a set number of client requests to the server.

      
    Note:
        - Vérifier sur le fichier FITS est compatible avec le modèle YOLO.
    """
    # Load configuration from JSON file
    config = load_config(os.path.join(CONFIGS_DIR,"param_cianna_rts_client.json"))
    
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
    image_folder = os.path.expanduser(config.get("IMAGE_FOLDER", "~/01_Observatoire/DIR_images"))

    print(f"Looking for images in {image_folder}...")
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

