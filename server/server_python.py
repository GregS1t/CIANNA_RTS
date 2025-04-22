
import os, shutil, subprocess

import uuid

from threading import Thread

from datetime import datetime
from werkzeug.utils import secure_filename
from flask import (Flask, request, jsonify, send_from_directory,
                   send_file, abort)

# Importing custom modules
from src.utils.job_logger import log_new_job, load_job_log, update_job_status
from src.processing.pipeline import run_prediction_job

#
# Flask app initialization
# This is the main server application for CIANNA.
# It handles the upload of XML and FITS files,
# processes them, and serves the CIANNA models.
# It also provides endpoints for uploading files,
# processing them, and moving them to different directories.
#
app = Flask(__name__, static_url_path='/models', static_folder='models')


# Directory definitions
__dirname = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(os.getcwd(), 'JOBS')
waiting_dir = os.path.join(base_dir, 'WAITING')
ongoing_dir = os.path.join(base_dir, 'ON_GOING')
completed_dir = os.path.join(base_dir, 'COMPLETED')
onerror_dir = os.path.join(base_dir, 'ON_ERROR')


# Configuration directories / programs
CONFIGS_DIR = os.path.join(os.getcwd(), 'configs')
# YOLO_CODE_DIR = os.path.join(__dirname, "src")
# YOLO_CODE_PATH = os.path.join(YOLO_CODE_DIR, "YC_RTS_detect.py")


# Create directories if they don't exist
for directory in [waiting_dir, ongoing_dir, completed_dir, onerror_dir]:
    os.makedirs(directory, exist_ok=True)


def get_profile_dir(root_dir, process_id):
    """
    Return the process-specific directory path for the given process_id.
    Create the directory if it does not exist.
    
    Args:
        root_dir (str): Base directory.
        process_id (str): Unique process identifier.
    
    Returns:
        str: Full path to the process-specific directory.
    """
    process_dir = os.path.join(root_dir, process_id)
    os.makedirs(process_dir, exist_ok=True)
    return process_dir

#
# Serve the CIANNA models XML file
#
@app.route('/models/<path:filename>')
def get_models(filename):
    """
    Serve the requested model file from the configs directory.
    Args:
        filename (str): The name of the file requested.
    Returns:
        File: The requested file if found.
    """
    return send_from_directory(CONFIGS_DIR, filename)


# Upload XML and FITS files from the client
# to the server and store them in the WAITING directory.
# The files are stored in WAITING/{profile_id} for processing.
# The profile_id is passed as a form parameter.
# Client function to call this endpoint:
#       def send_xml_fits_to_server(server_url, xml_data):

@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Endpoint to receive XML and FITS files.
    Files are stored in WAITING/{profile_id}.
    """

    # Check if XML file is missing.
    if 'xml' not in request.files:
        return jsonify({'message': 'SERVER: XML file is missing.'}), 400

    # Check if FITS file is missing.
    if 'fits' not in request.files:
        return jsonify({'message': 'SERVER: FITS file is missing.'}), 400

    # Generate a unique profile id on the server side.
    process_id = str(uuid.uuid4())
    process_dir = get_profile_dir(waiting_dir, process_id)

    xml_file = request.files['xml']
    fits_file = request.files['fits']

    xml_filename = secure_filename(request.files['xml'].filename)
    fits_filename = secure_filename(request.files['fits'].filename)

    xml_file.save(os.path.join(process_dir, xml_filename))
    fits_file.save(os.path.join(process_dir, fits_filename))

    xml_path = os.path.join(process_dir, xml_filename)
    fits_path = os.path.join(process_dir, fits_filename)

    # Log the job reception
    reception_date = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    log_new_job(process_id, fits_filename, reception_date)

    # Start processing in background
    Thread(target=run_prediction_job,
           args=(process_id, xml_path, fits_path, ongoing_dir,),
           ).start()

    return jsonify({'message': 'SERVER: Files have been uploaded. Job queued.',
                    'process_id': process_id}), 202



#
# Endpoint to check the status of a job
# and retrieve its metadata.
# The process_id is passed as a URL parameter.
# For example: 
#   curl http://localhost:3000/status/9a2e7b98-47a1-4d7a-bd17-98f447b57b26

@app.route('/status/<process_id>', methods=['GET'])
def get_job_status(process_id):
    """
    Return the current status and metadata of a given job.

    Args:
        process_id (str): Unique job identifier.

    Returns:
        JSON: Job metadata including status, comment, timestamps...
    """
    log_data = load_job_log()
    job = log_data.get(process_id)

    if job is None:
        return jsonify({
            'message': f'SERVER: No job found with process_id: {process_id}'
        }), 404

    return jsonify({
        'jobId': job.get("jobId"),
        'status': job.get("status"),
        'comment': job.get("comment"),
        'priority': job.get("priority"),
        'receptionDate': job.get("receptionDate"),
        'end_time': job.get("end_time"),
        'imagePath': job.get("imagePath"),
        'model': job.get("model"),
        'quantization': job.get("quantization")
    }), 200

#
# Endpoint to download the prediction result after processing
# The process_id is passed as a URL parameter.
#
#


@app.route('/download/<process_id>', methods=['GET'])
def download_result(process_id):
    """
    Endpoint to allow client to download prediction result for a given job.

    Args:
        process_id (str): The unique job identifier.

    Returns:
        The prediction file as attachment if the job is completed,
        otherwise a JSON error message.
    """
    log_data = load_job_log()
    job = log_data.get(process_id)

    if job is None:
        return jsonify({'message': f'No job found with id {process_id}'}), 404

    if job.get("status") != "COMPLETED":
        return jsonify({'message': f'Job {process_id} is not completed yet.'}), 400

    # Locate the prediction file
    fwd_res_dir = os.path.join(base_dir, 'COMPLETED', process_id, 'fwd_res')  # Or COMPLETED if you move files there
    filename = f"net0_rts_{process_id}.dat"  # adapt to your naming convention
    file_path = os.path.join(fwd_res_dir, filename)

    if not os.path.isfile(file_path):
        return jsonify({'message': f'Prediction file not found for job {process_id}.'}), 404

    try:
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({'message': f'Error sending file: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(port=3000, debug=True, host="0.0.0.0")
