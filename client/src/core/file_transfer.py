import os
import requests
import base64
import xml.etree.ElementTree as ET
from src.core.xml_utils import create_xml_param
from src.utils.tqdm_upfile import TqdmUploadFile

def send_xml_fits_to_server(server_url, xml_data):
    """
    Envoi un fichier XML et le fichier FITS associé sur le serveur.
    Le chemin de l'image est extrait du XML.
    """
    
    try:
        root = ET.fromstring(xml_data)
        image_path = root.find('Image/Path').text
        if not image_path or not os.path.exists(image_path):
            print("Invalid path to image in XML or file does not exist.")
            return None
    except ET.ParseError as e:
        print("XML parsing error:", e)
        return None

    try:
        file_size = os.path.getsize(image_path)
        with open(image_path, "rb") as f:
            wrapped_file = TqdmUploadFile(f, total=file_size, desc=f"Uploading {os.path.basename(image_path)}")
            files = {
                "xml": ("data.xml", xml_data, "application/xml"),
                "fits": ("image.fits", wrapped_file, "application/octet-stream")
            }
            response = requests.post(f"{server_url}/upload", files=files)
    except Exception as e:
        print("Error opening or sending file:", e)
        return None

    # Catch the response from the server
    if response.status_code == 202:
        process_id = response.json().get("process_id")
        print(response.json().get("message"))
        return process_id
    else:
        print("Sending error", response.text)
        return None



# Unused functions
def send_xml_only_to_server(server_url, xml_data):
    """
    Envoi uniquement le fichier XML au serveur.
    """
    headers = {'Content-Type': 'application/xml'}
    response = requests.post(f"{server_url}/upload", data=xml_data, headers=headers)
    if response.status_code == 202:
        return response.json().get("process_id")
    else:
        print("Erreur lors de l'envoi XML:", response.text)
        return None

def send_image_to_server(server_url, image_path):
    """
    Envoi une image (encodée en base64) au serveur.
    """
    try:
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Fichier image non trouvé: {image_path}")
        return None

    headers = {'Content-Type': 'application/json'}
    data = {"image": image_data}
    response = requests.post(f"{server_url}/upload_image", json=data, headers=headers)
    if response.status_code == 202:
        return response.json().get("process_id")
    else:
        print("Erreur lors de l'envoi de l'image:", response.text)
        return None

