
"""
CIANNA RTS Server - UWS Compliant 

This Flask application provides a Universal Worker Service (UWS) interface for
submitting and managing asynchronous jobs related to FITS images processing.
Jobs involve uploading XML configuration files and FITS images, processing them
with YOLO based models, and retrieving results.

Hereafter the list of the endpoints:

Routes:
    POST /jobs/:
        Create a new job by uploading XML and FITS files.

    GET /jobs/:
        List all existing jobs with their identifiers.

    GET /jobs/<job_id>:
        Retrieve the status and metadata of a specific job.

    GET /jobs/<job_id>/results:
        Download the result file (.dat) for a completed job.

    GET /models/<path:filename>:
        Serve model configuration files stored on the server.

Directory Structure:
    JOBS/
        PENDING/    - Jobs awaiting execution
        EXECUTING/  - Jobs currently in execution
        COMPLETED/  - Jobs successfully completed
        ERROR/      - Jobs that encountered an error

Each job has a unique ID and is stored in its respective phase directory.

The server automatically adapts response format based on the 'Accept' HTTP header,
returning either JSON (default) or XML compliant with UWS 1.0 standards.

"""


import os, shutil, subprocess, sys
import argparse
import uuid
from threading import Thread
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import (Flask, request, jsonify, send_from_directory,
                   send_file, abort, Response, redirect)

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


# Directory definitions -> redefined to follow UWS pattern
__dirname = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(os.getcwd(), 'JOBS')
PENDING_DIR = os.path.join(BASE_DIR, 'PENDING') # former WAITING
EXECUTING_DIR = os.path.join(BASE_DIR, 'EXECUTING') # former ON_GOING
COMPLETED_DIR = os.path.join(BASE_DIR, 'COMPLETED') 
ERROR_DIR = os.path.join(BASE_DIR, 'ERROR')
CONFIGS_DIR = os.path.join(os.getcwd(), 'configs')

# Create directories if needed
for d in [PENDING_DIR, EXECUTING_DIR, COMPLETED_DIR, ERROR_DIR]:
    os.makedirs(d, exist_ok=True)

# Few helper functions
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

def get_job_directory(job_id):
    """Retrieve the directory and current phase for a given job ID.

    Args:
        job_id (str): Unique identifier of the job.

    Returns:
        tuple: (job directory path, current phase name) or (None, None) if not found.
    """
    for state_dir in [PENDING_DIR, EXECUTING_DIR, COMPLETED_DIR, ERROR_DIR]:
        job_path = os.path.join(state_dir, job_id)
        if os.path.isdir(job_path):
            return job_path, os.path.basename(state_dir)
    return None, None

def determine_phase(state_folder_name):
    """Map a directory name to its corresponding UWS phase.

    Args:
        state_folder_name (str): Name of the folder representing job phase.

    Returns:
        str: UWS phase corresponding to the folder name.
    """
    mapping = {
        'PENDING': 'PENDING',
        'EXECUTING': 'EXECUTING',
        'COMPLETED': 'COMPLETED',
        'ERROR': 'ERROR'
    }
    return mapping.get(state_folder_name, 'UNKNOWN')

def generate_xml_response(content_dict):
    """Generate a UWS-compliant XML response from a dictionary.

    Args:
        content_dict (dict): Dictionary containing job metadata.

    Returns:
        str: XML formatted string representing the job information.
    """
    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<uws:job xmlns:uws="http://www.ivoa.net/xml/UWS/v1.0">')
    for key, value in content_dict.items():
        xml.append(f'  <uws:{key}>{value}</uws:{key}>')
    xml.append('</uws:job>')
    return '\n'.join(xml)



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

#
# Endpoint to create a new job / UWS compliant
#
@app.route('/jobs/', methods=['POST'])

def create_job():
    """Create a new asynchronous job by uploading XML and FITS files.

    Returns:
        Response: HTTP 303 redirecting to the newly created job status.
    """
    if 'xml' not in request.files or 'fits' not in request.files:
        return jsonify({'error': 'Missing XML or FITS file.'}), 400

    process_id = str(uuid.uuid4())
    process_dir = os.path.join(PENDING_DIR, process_id)
    os.makedirs(process_dir, exist_ok=True)

    xml_file = request.files['xml']
    fits_file = request.files['fits']
    xml_path = os.path.join(process_dir, secure_filename(xml_file.filename))
    fits_path = os.path.join(process_dir, secure_filename(fits_file.filename))

    xml_file.save(xml_path)
    fits_file.save(fits_path)

    reception_date = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    log_new_job(process_id, fits_file.filename, reception_date)

    Thread(target=run_prediction_job, args=(process_id, xml_path, fits_path,
                                            EXECUTING_DIR,)).start()

    location_url = f"/jobs/{process_id}"

    # Return a proper 303 response manually
    response = Response(status=303)
    response.headers['Location'] = location_url
    return response
#
# List of all jobs
#
@app.route('/jobs/', methods=['GET'])
def list_jobs():
    """List all existing jobs available on the server.

    Returns:
        Response: A list of jobs in JSON or XML format based on Accept header.
    """
    job_ids = []
    for state_dir in [PENDING_DIR, EXECUTING_DIR, COMPLETED_DIR, ERROR_DIR]:
        job_ids.extend(os.listdir(state_dir))

    accept = request.headers.get('Accept', '')
    if 'application/xml' in accept:
        xml = ['<?xml version="1.0" encoding="UTF-8"?>', '<uws:jobs xmlns:uws="http://www.ivoa.net/xml/UWS/v1.0">']
        for jid in job_ids:
            xml.append(f'  <uws:jobref id="{jid}" href="/jobs/{jid}"/>')
        xml.append('</uws:jobs>')
        return Response('\n'.join(xml), mimetype='application/xml')
    else:
        return jsonify({'jobs': job_ids})

#
# Endpoint to retrieve the status and metadata of a specific job
#
@app.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Retrieve the status and metadata of a specific job.

    Args:
        job_id (str): Unique identifier of the job.

    Returns:
        Response: Job metadata in JSON or XML format.
    """
    job_dir, state_folder = get_job_directory(job_id)
    if job_dir is None:
        return jsonify({'error': 'Job not found.'}), 404

    phase = determine_phase(state_folder)

    job_info = {
        'jobId': job_id,
        'phase': phase,
        'timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    }

    accept = request.headers.get('Accept', '')
    if 'application/xml' in accept:
        return Response(generate_xml_response(job_info), mimetype='application/xml')
    else:
        return jsonify(job_info)


@app.route('/jobs/<job_id>/results', methods=['GET'])
def get_job_results(job_id):
    """Retrieve the output file (.dat) for a completed job.

    Args:
        job_id (str): Unique identifier of the job.

    Returns:
        Response: File download or error message.
    """
    job_dir, state_folder = get_job_directory(job_id)
    if job_dir is None:
        return jsonify({'error': 'Job not found.'}), 404

    if state_folder != 'COMPLETED':
        return jsonify({'error': f'Job not completed. Current state: {state_folder}'}), 400

    fwd_res_dir = os.path.join(job_dir, 'fwd_res')
    expected_file = f"net0_rts_{job_id}.dat"
    result_path = os.path.join(fwd_res_dir, expected_file)

    if not os.path.isfile(result_path):
        return jsonify({'error': 'Result file not found.'}), 404

    return send_file(result_path, as_attachment=True)



# Upload XML and FITS files from the client
# to the server and store them in the WAITING directory.
# The files are stored in WAITING/{profile_id} for processing.
# The profile_id is passed as a form parameter.
# Client function to call this endpoint:
#       def send_xml_fits_to_server(server_url, xml_data):

# @app.route('/upload', methods=['POST'])
# def upload_files():
#     """
#     Endpoint to receive XML and FITS files.
#     Files are stored in WAITING/{profile_id}.
#     """

#     # Check if XML file is missing.
#     if 'xml' not in request.files:
#         return jsonify({'message': 'SERVER: XML file is missing.'}), 400

#     # Check if FITS file is missing.
#     if 'fits' not in request.files:
#         return jsonify({'message': 'SERVER: FITS file is missing.'}), 400

#     # Generate a unique profile id on the server side.
#     process_id = str(uuid.uuid4())
#     process_dir = get_profile_dir(waiting_dir, process_id)

#     xml_file = request.files['xml']
#     fits_file = request.files['fits']

#     xml_filename = secure_filename(request.files['xml'].filename)
#     fits_filename = secure_filename(request.files['fits'].filename)

#     xml_file.save(os.path.join(process_dir, xml_filename))
#     fits_file.save(os.path.join(process_dir, fits_filename))

#     xml_path = os.path.join(process_dir, xml_filename)
#     fits_path = os.path.join(process_dir, fits_filename)

#     # Log the job reception
#     reception_date = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
#     log_new_job(process_id, fits_filename, reception_date)

#     # Start processing in background
#     Thread(target=run_prediction_job,
#            args=(process_id, xml_path, fits_path, ongoing_dir,),
#            ).start()

#     return jsonify({'message': 'SERVER: Files have been uploaded. Job queued.',
#                     'process_id': process_id}), 202


#
# Endpoint to check the status of a job
# and retrieve its metadata.
# The process_id is passed as a URL parameter.
# For example: 
#   curl http://localhost:3000/status/9a2e7b98-47a1-4d7a-bd17-98f447b57b26

# @app.route('/status/<process_id>', methods=['GET'])
# def get_job_status(process_id):
#     """
#     Return the current status and metadata of a given job.

#     Args:
#         process_id (str): Unique job identifier.

#     Returns:
#         JSON: Job metadata including status, comment, timestamps...
#     """
#     log_data = load_job_log()
#     job = log_data.get(process_id)

#     if job is None:
#         return jsonify({
#             'message': f'SERVER: No job found with process_id: {process_id}'
#         }), 404

#     return jsonify({
#         'jobId': job.get("jobId"),
#         'status': job.get("status"),
#         'comment': job.get("comment"),
#         'priority': job.get("priority"),
#         'receptionDate': job.get("receptionDate"),
#         'end_time': job.get("end_time"),
#         'imagePath': job.get("imagePath"),
#         'model': job.get("model"),
#         'quantization': job.get("quantization")
#     }), 200

#
# Endpoint to download the prediction result after processing
# The process_id is passed as a URL parameter.
#
#


# @app.route('/download/<process_id>', methods=['GET'])
# def download_result(process_id):
#     """
#     Endpoint to allow client to download prediction result for a given job.

#     Args:
#         process_id (str): The unique job identifier.

#     Returns:
#         The prediction file as attachment if the job is completed,
#         otherwise a JSON error message.
#     """
#     log_data = load_job_log()
#     job = log_data.get(process_id)

#     if job is None:
#         return jsonify({'message': f'No job found with id {process_id}'}), 404

#     if job.get("status") != "COMPLETED":
#         return jsonify({'message': f'Job {process_id} is not completed yet.'}), 400

#     # Locate the prediction file
#     fwd_res_dir = os.path.join(base_dir, 'COMPLETED', process_id, 'fwd_res')  # Or COMPLETED if you move files there
#     filename = f"net0_rts_{process_id}.dat"  # adapt to your naming convention
#     file_path = os.path.join(fwd_res_dir, filename)

#     if not os.path.isfile(file_path):
#         return jsonify({'message': f'Prediction file not found for job {process_id}.'}), 404

#     try:
#         return send_file(file_path, as_attachment=True)
#     except Exception as e:
#         return jsonify({'message': f'Error sending file: {str(e)}'}), 500


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=3000, help='Port to run the Flask server on')
    args = parser.parse_args()

    app.run(port=args.port, debug=True, host="0.0.0.0")
