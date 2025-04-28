import requests
import xml.etree.ElementTree as ET
import os, sys

def download_xml(url):
    """
    Download the XML content from the specified URL.

    Args:
        url (str): The URL from which to download the XML content.

    Returns:
        str or None: The XML content as a string if successful, otherwise None.
    """
    try:
        print("URL: {}".format(url))
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print("Error during download:", e)
        return None

def update_cianna_models(url, local_file):
    """
    Download the remote XML file containing CIANNA models and update the local file,
    always retrieving the latest version without performing any version check.

    Args:
        url (str): The URL of the remote XML file.
        local_file (str): The path to the local file where the XML should be saved.

    Returns:
        bool or None: Returns True if the local file was updated successfully,
                      or None if the update failed.
    """
    print("Downloading remote file...")
    print("URL: {}".format(url))
    print("Local file: {}".format(local_file))

    remote_xml = download_xml(url)
    if remote_xml is None:
        print("Failed to download remote file.")
        return None

    if local_file is not None:
        # Ensure that the local directory exists
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        # Write the downloaded content to the local file, overwriting any existing file
        with open(local_file, 'w', encoding='utf-8') as f:
            f.write(remote_xml)
        print("The local file has been updated to the latest version.")
        return True
    else:
        sys.exit("Error: Unable to load the local file.")

def get_model_info(xml_path, model_id):
    """
    Parses a CIANNA_models.xml file and retrieves the model information
    for a given model ID.

    Parameters
    ----------
    xml_path : str
        Path to the XML file that contains CIANNA model definitions.
    model_id : str
        The identifier of the model to retrieve (from the 'id' attribute in
        the <Model> tag).

    Returns
    -------
    dict or None
        A dictionary containing the model's parameters if found,
        otherwise None.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for model in root.findall("Model"):
        if model.attrib.get("id") == model_id:
            info = {
                "Name": model.findtext("Name"),
                "ReleaseDate": model.findtext("ReleaseDate"),
                "OriginalInputDim": model.findtext("OriginalInputDim"),
                "MinInputDim": model.findtext("MinInputDim"),
                "MaxInputDim": model.findtext("MaxInputDim"),
                "YOLOGridElemDim": model.findtext("YOLOGridElemDim"),
                "DataNormalization": model.findtext("DataNormalization"),
                "DataQuantization": model.findtext("DataQuantization"),
                "TrainingQuantization": model.findtext("TrainingQuantization"),
                "InferenceQuantization": model.findtext("InferenceQuantization"),
                "InferenceMode": model.findtext("InferenceMode"),
                "InferencePatchShift": model.findtext("InferencePatchShift"),
                "ReceptiveField": model.findtext("ReceptiveField"),
                "CheckpointPath": model.findtext("CheckpointPath"),
                "Comments": model.findtext("Comments"),
            }
            return info
    return None 