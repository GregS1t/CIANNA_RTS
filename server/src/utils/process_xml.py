import sys
import xml.etree.ElementTree as ET
import base64
import csv
from datetime import datetime

def process_xml(input_file, output_csv):
    try:
        # Check if the input file is empty
        with open(input_file, 'r') as f:
            if not f.read().strip():
                raise ValueError("The input file is empty or invalid.")
        
        # Parse the XML file
        tree = ET.parse(input_file)
        root = tree.getroot()

        # Extract fields with fallback for missing/empty tags
        user_id = root.find('USER_ID').text if root.find('USER_ID') is not None else 'N/A'
        timestamp = root.find('Timestamp').text if root.find('Timestamp') is not None else 'N/A'

        coordinates = root.find('Coordinates')
        ra = coordinates.find('RA').text if coordinates is not None and coordinates.find('RA') is not None else 'N/A'
        dec = coordinates.find('DEC').text if coordinates is not None and coordinates.find('DEC') is not None else 'N/A'
        h = coordinates.find('H').text if coordinates is not None and coordinates.find('H') is not None else 'N/A'
        w = coordinates.find('W').text if coordinates is not None and coordinates.find('W') is not None else 'N/A'

        image = root.find('Image').text if root.find('Image') is not None else 'N/A'
        quantization = root.find('Quantization').text if root.find('Quantization') is not None else 'N/A'

        # Write to CSV
        with open(output_csv, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['USER_ID', 'Timestamp', 'RA', 'DEC', 'H', 'W', 'Image', 'Quantization', 'Status'])
            writer.writerow([user_id, timestamp, ra, dec, h, w, image, quantization, 'COMPLETED'])

        print("File processed successfully.")

    except ET.ParseError as e:
        print(f"Error while processing the XML file: {e}")
        with open(output_csv, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Status'])
            writer.writerow(['ERROR'])

    except ValueError as e:
        print(f"Error: {e}")
        with open(output_csv, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Status'])
            writer.writerow(['ERROR'])

    except Exception as e:
        print(f"Unexpected error: {e}")
        with open(output_csv, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Status'])
            writer.writerow(['ERROR'])


def save_error_to_csv(output_csv, status):
    """
    Temporary function to handle error. 
    The future implementation should include more details about the error during the YOLO-CIANNA processing.
    """
    with open(output_csv, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", status])

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 process_xml.py <input_file.xml> <output_file.csv>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        print("Processing XML file:", input_file)
        # # Check if output file is exists
        # try:
        #     with open(output_file, mode="x", newline="") as csv_file:
        #         pass

        # except FileExistsError: 
        #     print(f"Output file '{output_file}' already exists. Please provide a new file name.")
        #     sys.exit(1)


        # Add headers to the CSV if it's empty
        try:
            with open(output_file, mode="x", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["USER_ID", "Timestamp", "RA", "DEC", "H", "W", "Quantization", "Image_File", "STATUS"])
        except FileExistsError:
            pass  # The file already exists, skip header writing
        
        process_xml(input_file, output_file)