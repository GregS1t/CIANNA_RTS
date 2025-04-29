import os
import requests
import base64
import xml.etree.ElementTree as ET
from src.core.xml_utils import create_xml_param
from src.utils.tqdm_upfile import TqdmUploadFile

def send_xml_fits_to_server(server_url, xml_data):
    """
    Send XML and FITS file to the server under UWS-compliant /jobs/ endpoint.

    Args:
        server_url (str): Server base URL.
        xml_data (str): XML parameters as string.

    Returns:
        str: Job ID returned by the server, or None if failed.
    """
    try:
        root = ET.fromstring(xml_data)
        image_path = root.find('Image/Path').text
        if not image_path or not os.path.exists(image_path):
            print("[CLIENT] Invalid image path or file does not exist.")
            return None
    except ET.ParseError as e:
        print("[CLIENT] XML parsing error:", e)
        return None

    try:
        file_size = os.path.getsize(image_path)
        with open(image_path, "rb") as f:
            wrapped_file = TqdmUploadFile(f, total=file_size,
                                          desc=f"Uploading {os.path.basename(image_path)}")
            files = {
                "xml": ("parameters.xml", xml_data, "application/xml"),
                "fits": ("image.fits", wrapped_file, "application/octet-stream")
            }
            response = requests.post(f"{server_url}/jobs/", files=files,
                                     allow_redirects=False)
    except Exception as e:
        print("[CLIENT] Error opening or sending file:", e)
        return None

    if response.status_code == 303:
        location = response.headers.get('Location')
        if location:
            job_id = location.rstrip('/').split('/')[-1]
            print("[CLIENT] Job submitted successfully.")
            return job_id
    else:
        print("[CLIENT] Sending error", response.text)
        return None
