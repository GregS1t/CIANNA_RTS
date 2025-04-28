import os, json
import shutil
import numpy as np
from datetime import datetime
from astropy.io import fits
from astropy.wcs import WCS
from xml.etree import ElementTree as ET

from src.utils.data_gen import *
from src.utils.job_logger import update_job_status, load_job_log

#
# GET INFO FROM CONFIG FILE
# 

PARAMS_PATH = os.path.join(os.path.dirname(__file__), '../../params/yolo_rts.json')
with open(PARAMS_PATH, 'r') as f:
    CONFIG = json.load(f)

sys.path.insert(0, CONFIG["PATH2CIANNA"]) # To declare CIANNA path

SERVER_DIR = CONFIG.get("SERVER_DIR", os.path.dirname(__file__))

MODEL_ROOT = CONFIG.get("CIANNA_RTS_DIR", ".")
MODEL_DIR = CONFIG.get("model_dir", os.path.join(MODEL_ROOT, "net_save"))

JOBS_CONFIG = CONFIG.get("JOBS", {})

JOBS_WAITING = os.path.join(SERVER_DIR,
                            JOBS_CONFIG.get("WAITING",
                                            os.path.join(MODEL_ROOT,
                                                         "data",
                                                         "WAITING")))
JOBS_ON_GOING = os.path.join(SERVER_DIR,
                             JOBS_CONFIG.get("ON_GOING",
                                             os.path.join(MODEL_ROOT,
                                                          "data",
                                                          "ON_GOING")))
JOBS_COMPLETED = os.path.join(SERVER_DIR,
                              JOBS_CONFIG.get("COMPLETED", os.path.join(MODEL_ROOT,
                                                           "data",
                                                           "COMPLETED")))


def prepare_job_directory(base_output_dir, process_id):
    """
    Prepare the ON_GOING directory for a job and create subfolders if needed.

    Args:
        base_output_dir (str): Root output folder (e.g., ON_GOING).
        process_id (str): Unique identifier for the job.

    Returns:
        str: Path to the job directory.
    """
    job_dir = os.path.join(base_output_dir, process_id)
    os.makedirs(job_dir, exist_ok=True)

    fwd_res_dir = os.path.join(job_dir, "fwd_res")
    os.makedirs(fwd_res_dir, exist_ok=True)

    print(f"[DIR] ON_GOING directory prepared at: {job_dir}")
    return job_dir

def parse_job_parameters(xml_path):
    """
    Parse the XML file and extract job parameters.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    user_id = root.find('USER_ID').text
    timestamp = root.find('Timestamp').text

    coords = root.find('Coordinates')
    coordinates = {
        'RA': float(coords.find('RA').text),
        'DEC': float(coords.find('DEC').text),
        'H': int(coords.find('H').text),
        'W': int(coords.find('W').text)
    }

    image_path = root.find('Image/Path').text
    model = root.find('YOLO_Model/Name').text
    quantization = root.find('Quantization').text

    print(f"[XML] Parsed: user_id={user_id}, model={model}, image={image_path}")

    return {
        'user_id': user_id,
        'timestamp': timestamp,
        'coordinates': coordinates,
        'image_path': image_path,
        'model': model,
        'quantization': quantization,
        'fits_filename': os.path.basename(image_path)  # used for local path inside ON_GOING
    }

def validate_model_path(model_name, path2models_root):
    """
    Construct and verify the full path to the model.

    """
    model_path = os.path.join(path2models_root, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"[MODEL] Model found at: {model_path}")
    return model_path


def run_prediction(process_id, job_dir, fits_file, model_path, params):
    import CIANNA as cnn

    os.chdir(job_dir)
    print(f"[RUN] Working directory set to {job_dir}")

    coordinates = [
        params['coordinates']['RA'],
        params['coordinates']['DEC'],
        params['coordinates']['H'],
        params['coordinates']['W']
    ]

    fit_name_original = os.path.basename(fits_file)
    fwd_res_dir = os.path.join(job_dir, 'fwd_res')

    # Model setup
    outdim = 0
    b_size = 8

    cnn.init(
        in_dim=i_ar([image_size, image_size]), in_nb_ch=1,
        out_dim=outdim, bias=0.1, b_size=b_size, comp_meth='C_CUDA',
        inference_only=1, dynamic_load=1,
        mixed_precision=params['quantization'], adv_size=35
    )
    cnn.set_yolo_params()
    cnn.load(model_path, 0, bin=1)

    # FITS reading
    hdul = fits.open(fits_file)
    full_img = hdul[0].data[0, 0]
    hdul.close()

    min_pix = np.percentile(full_img, 99.4)
    max_pix = np.percentile(full_img, 99.8)
    full_data_norm = np.clip(full_img, min_pix, max_pix)
    full_data_norm = (full_data_norm - min_pix) / (max_pix - min_pix)
    full_data_norm = np.tanh(3.0 * full_data_norm)

    map_pixel_size = full_data_norm.shape[0]
    size_px = map_pixel_size
    size_py = full_data_norm.shape[1]

    nb_area_w = int((map_pixel_size - 128) / 240) + 1
    nb_area_h = int((map_pixel_size - 128) / 240) + 1
    nb_images_all = nb_area_w * nb_area_h

    input_data = np.zeros((nb_images_all, image_size * image_size),
                          dtype="float32")
    patch = np.zeros((image_size, image_size), dtype="float32")

    for i_d in range(nb_images_all):
        p_y = i_d // nb_area_w
        p_x = i_d % nb_area_w

        xmin = p_x * 240 - 128
        xmax = xmin + image_size
        ymin = p_y * 240 - 128
        ymax = ymin + image_size

        px_min, px_max = 0, image_size
        py_min, py_max = 0, image_size

        set_zero = False
        if xmin < 0:
            px_min = -xmin
            xmin = 0
            set_zero = True
        if ymin < 0:
            py_min = -ymin
            ymin = 0
            set_zero = True
        if xmax > size_px:
            px_max = image_size - (xmax - size_px)
            xmax = size_px
            set_zero = True
        if ymax > size_py:
            py_max = image_size - (ymax - size_py)
            ymax = size_py
            set_zero = True

        if set_zero:
            patch[:, :] = 0.0

        patch[px_min:px_max, py_min:py_max] = np.flip(full_data_norm[xmin:xmax, ymin:ymax], axis=0)
        input_data[i_d, :] = patch.flatten("C")

    targets = np.zeros((nb_images_all, outdim), dtype="float32")
    cnn.create_dataset("TEST", nb_images_all, input_data, targets)
    cnn.forward(repeat=1, no_error=1, saving=2, drop_mode="AVG_MODEL")

    pred_file = os.path.join(fwd_res_dir, f"net0_rts_{process_id}.dat")

    if os.path.exists(os.path.join(fwd_res_dir, "net0_0000.dat")):
        os.rename(os.path.join(fwd_res_dir, "net0_0000.dat"), pred_file)
        print("[RUN] Prediction file saved.")
    else:
        raise FileNotFoundError("Prediction output file 'net0_0000.dat' not found.")

    cnn.delete_dataset("TEST")
    return pred_file

#
# Function to prepare the job directory
#
def run_prediction_job(process_id, xml_path, fits_path, 
                       model_root=MODEL_ROOT, model_dir=MODEL_DIR):

    """
    Main pipeline to run a prediction for a given job.

    Args:
        process_id (str): Unique job identifier.
        xml_path (str): Path to the XML configuration file.
        fits_path (str): Path to the FITS image.
        base_output_dir (str): Root output directory for ON_GOING jobs.
    """
    print(f"[PIPELINE] Starting job {process_id}")

    try:
        # Move full job directory from WAITING to ON_GOING
        waiting_job_dir = os.path.join(JOBS_WAITING, process_id)
        ongoing_job_dir = os.path.join(JOBS_ON_GOING, process_id)
        shutil.move(waiting_job_dir, ongoing_job_dir)
        print(f"[PIPELINE] Moved {waiting_job_dir} to {ongoing_job_dir}")

        # Use the new location as the job directory
        params = parse_job_parameters(os.path.join(ongoing_job_dir, os.path.basename(xml_path)))
        model_path = validate_model_path(params['model'], model_dir)
        print(f"[PIPELINE] output_root: {JOBS_ON_GOING},\n job_dir: {ongoing_job_dir}")

        prediction_file = run_prediction(
            process_id,
            ongoing_job_dir,
            fits_file=os.path.join(ongoing_job_dir, "image.fits"), # All the images are named "image.fits"
            model_path=model_path,
            params=params
        )

        # Update status to COMPLETED
        update_job_status(
            process_id,
            status="COMPLETED",
            comment="Prediction completed",
            end_time=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        )

        # Move job directory to COMPLETED
        completed_dir = os.path.abspath(os.path.join(JOBS_COMPLETED, process_id))
        shutil.move(ongoing_job_dir, completed_dir)
        print(f"[PIPELINE] Job {process_id} moved to COMPLETED.")
        return os.path.join(completed_dir, 'fwd_res', f"net0_rts_{process_id}.dat")

    except Exception as e:
        update_job_status(
            process_id,
            status="ERROR",
            comment=str(e),
            end_time=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        )
        print(f"[PIPELINE] Job {process_id} failed: {e}")