import xml.etree.ElementTree as ET
from datetime import datetime

def create_xml_param(user_id, ra, dec, h, w, image_path, yolo_model, quantization):
    """
    Crée une structure XML contenant les paramètres de la requête.
    """
    root = ET.Element("YOLO_CIANNA")
    
    # Informations utilisateur et date
    ET.SubElement(root, "USER_ID").text = str(user_id)
    ET.SubElement(root, "Timestamp").text = datetime.now().isoformat()
    
    # Coordonnées
    coords = ET.SubElement(root, "Coordinates")
    ET.SubElement(coords, "RA").text = str(ra)
    ET.SubElement(coords, "DEC").text = str(dec)
    ET.SubElement(coords, "H").text = str(h)
    ET.SubElement(coords, "W").text = str(w)
    
    # Chemin de l'image
    image_elem = ET.SubElement(root, "Image")
    ET.SubElement(image_elem, "Path").text = image_path
    
    # Modèle YOLO et quantization
    yolo_elem = ET.SubElement(root, "YOLO_Model")
    ET.SubElement(yolo_elem, "Name").text = yolo_model
    ET.SubElement(root, "Quantization").text = str(quantization)
    
    return ET.tostring(root, encoding="utf-8", method="xml")
