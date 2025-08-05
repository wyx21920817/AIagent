import requests
import os
import json
import pydicom
from io import BytesIO
import numpy as np
from fastapi import Request
import logging
import traceback
import requests
import os
import pydicom
from typing import List, Dict, Optional
from app.utils.auth_utils import get_temp_token

# Get Orthanc server URL from configuration
ORTHANC_SERVER_URL = os.getenv('ORTHANC_SERVER_URL', 'http://127.0.0.1:8042')

def get_headers(token):
    """
    Get request headers with JWT token
    
    Args:
        token (str): JWT token
        
    Returns:
        dict: Dictionary containing Authorization header
    """
    if not token:
        raise ValueError("Token is required for Orthanc API calls")
    
    print(f"Using token for Orthanc API: {token}")  # Debug token
    return {
        'Authorization': f"Bearer {token}",
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }

def upload_dicom_file(file_path, token):
    """
    Upload a DICOM file to the Orthanc server.
    
    Args:
        file_path (str): The path to the DICOM file to upload
        token (str): JWT token for authentication
        
    Returns:
        dict: Response from the server containing the ID of the uploaded instance
        or None if upload fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DICOM file not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            dicom_content = f.read()
            
        # Upload the DICOM file to Orthanc
        headers = get_headers(token)
        headers['Content-Type'] = 'application/dicom'
        
        print(f"Sending request to Orthanc with headers: {headers}")  # Debug headers
        
        response = requests.post(
            f"{ORTHANC_SERVER_URL}/instances",
            headers=headers,
            data=dicom_content,
            verify=False,  # May be needed for self-signed certificates
            timeout=(5, 30)  # Connection and read timeout
        )
        
        # Log response details
        
        if response.status_code in [200, 201]:  # Accept 200 or 201
            return response.json()
        else:
            error_msg = f"Error uploading DICOM file: {response.status_code}. Response: {response.text}"
            return {"error": error_msg}
            
    except Exception as e:
        error_msg = f"Error uploading DICOM file: {str(e)}"
        return {"error": error_msg}

def upload_dicom_folder(folder_path, token):
    """
    Upload all DICOM files from a folder to the Orthanc server.
    
    Args:
        folder_path (str): Path to the folder containing DICOM files
        token (str): JWT token for authentication
        
    Returns:
        list: List of successful uploads (instance IDs)
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
        
    successful_uploads = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.dcm', '.dicom')):
                file_path = os.path.join(root, file)
                result = upload_dicom_file(file_path, token)
                if result:
                    successful_uploads.append(result)
                    
    return successful_uploads

def get_instance(instance_id: str, token: str):
    """
    Get DICOM instance details from Orthanc server.
    
    Args:
        instance_id (str): The Orthanc instance ID
        token (str): JWT token for authentication
        
    Returns:
        dict: Instance details or None if not found
    """
    try:
        
        response = requests.get(
            f"{ORTHANC_SERVER_URL}/instances/{instance_id}",
            headers=get_headers(token)
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting instance: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error getting instance: {str(e)}")
        return None

def get_instance_dicom(instance_id, token):
    """
    Get DICOM file data from Orthanc server.
    
    Args:
        instance_id (str): The Orthanc instance ID
        token (str): JWT token for authentication
        
    Returns:
        bytes: Raw DICOM data or None if not found
    """
    try:
        
        # First try with current token
        response = requests.get(
            f"{ORTHANC_SERVER_URL}/instances/{instance_id}/file",
            headers=get_headers(token)
        )
        
        # If unauthorized, try with a new token
        if response.status_code == 403:
            print("Token expired, getting new token...")
            token = get_temp_token(force_new=True)
            response = requests.get(
                f"{ORTHANC_SERVER_URL}/instances/{instance_id}/file",
                headers=get_headers(token)
            )
        
        if response.status_code == 200:
            return response.content
        else:
            print(f"Error getting DICOM file: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error getting DICOM file: {str(e)}")
        return None

def modify_instance_metadata(instance_id, metadata, token):
    """
    Modify DICOM tags of an instance in Orthanc server using the /modify endpoint.
    Creates a new instance with modified tags while keeping the original instance unchanged.
    
    Args:
        instance_id (str): The Orthanc instance ID
        metadata (dict): Dictionary containing metadata (notes and tags)
        token (str): JWT token for authentication
        
    Returns:
        str: New instance ID if successful, None otherwise
    """
    try:
        headers = get_headers(token)
        headers['Content-Type'] = 'application/json'
        
        # Use private tags to store custom information
        dicom_tags = {
           
            "0009,1010": metadata.get('Notes', ''),  # Private tag for storing Notes
            "0009,1011": metadata.get('Tags', ''),   # Private tag for storing Tags
            "0020,4000": metadata.get('Notes', '')   # Also store in ImageComments for compatibility
        }
        
        url = f"{ORTHANC_SERVER_URL}/instances/{instance_id}/modify"
        print(f"Modifying instance tags at: {url}")
        print(f"Headers: {headers}")
        print(f"DICOM tags to modify: {dicom_tags}")
        
        request_data = {
            "Replace": dicom_tags,
            "Force": True,
            "KeepSource": True,
            "PrivateCreator": "WebServer", 
        }
        
        # Check instance list before modification
        instances_url = f"{ORTHANC_SERVER_URL}/instances"
        instances_response = requests.get(instances_url, headers=headers)
        if instances_response.status_code == 200:
            print(f"Current instances before modify: {instances_response.json()}")
        
        response = requests.post(
            url,
            headers=headers,
            json=request_data
        )
        
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        print(f"Response content length: {len(response.content)}")
        print(f"First 200 bytes of response: {response.content[:200]}")
        
        # Check instance list after modification
        instances_response = requests.get(instances_url, headers=headers)
        if instances_response.status_code == 200:
            print(f"Current instances after modify: {instances_response.json()}")
        
        # Check if modification was successful
        if response.status_code != 200:
            print(f"Failed to modify instance metadata: {response.status_code}")
            return None
            
        print("Successfully modified instance metadata, uploading modified instance...")
        
        # Try to upload returned DICOM data as new instance
        upload_url = f"{ORTHANC_SERVER_URL}/instances"
        upload_headers = headers.copy()
        upload_headers['Content-Type'] = 'application/dicom'
        upload_response = requests.post(
            upload_url,
            headers=upload_headers,
            data=response.content
        )
        
        if upload_response.status_code != 200:
            print(f"Failed to upload modified instance: {upload_response.status_code}")
            return None
            
        print(f"Upload response content: {upload_response.json()}")
        
        # Get current list of all instances
        instances_response = requests.get(instances_url, headers=headers)
        current_instances = []
        if instances_response.status_code == 200:
            current_instances = instances_response.json()
            print(f"Current instances after upload: {current_instances}")
            
        try:
            result = upload_response.json()
            new_instance_id = result.get('ID')
            if new_instance_id:
                print(f"Successfully created new instance: {new_instance_id}")
                return {
                    'instance_id': new_instance_id,
                    'details': f"Modify instance: {instance_id}",
                    'current_instances': current_instances
                }
        except Exception as e:
            print(f"Error parsing upload response: {str(e)}")
            return None
        
    except Exception as e:
        print(f"Error modifying instance tags: {str(e)}")
        import traceback
        print(f"Full error traceback: {traceback.format_exc()}")
        return None


def get_patient_studies(patient_id: str, token: str):
    """
    Get all studies for a specific patient.
    
    Args:
        patient_id (str): The patient ID
        token (str): JWT token for authentication
        
    Returns:
        list: List of study details
    """
    try:
        
        headers = get_headers(token)
        
        # Get patient information directly
        response = requests.get(
            f"{ORTHANC_SERVER_URL}/patients/{patient_id}",
            headers=headers
        )
        
        if response.status_code != 200:
            print(f"Error getting patient: {response.status_code}")
            print(f"Response: {response.text}")
            return []
            
        patient_data = response.json()
        studies = patient_data.get('Studies', [])
        study_details = []
        
        # Get detailed information for each study
        for study_id in studies:
            study_response = requests.get(
                f"{ORTHANC_SERVER_URL}/studies/{study_id}",
                headers=headers
            )
            if study_response.status_code == 200:
                study_details.append(study_response.json())
        
            return study_details
            
    except Exception as e:
        print(f"Error getting patient studies: {str(e)}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return []

def get_study_instances(study_id: str, token: str):
    """
    Get all instances in a study.
    
    Args:
        study_id (str): The study ID
        token (str): JWT token for authentication
        
    Returns:
        list: List of instance details
    """
    try:
        
        
        headers = get_headers(token)
        
        response = requests.get(
            f"{ORTHANC_SERVER_URL}/studies/{study_id}/instances",
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting study instances: {response.status_code}")
            print(f"Response: {response.text}")
            return []
            
    except Exception as e:
        print(f"Error getting study instances: {str(e)}")
        return []

def get_patient_instances(patient_id: str, token: str):
    """
    Get all instances for a specific patient.
    
    Args:
        patient_id (str): The patient ID
        token (str): JWT token for authentication
        
    Returns:
        list: List of instance details
    """
    try:
        headers = get_headers(token)
        print(f"[Orthanc] Getting patient info for ID: {patient_id}")
        print(f"[Orthanc] Using headers: {headers}")
        
        # First, search for the patient by ID
        patients = search_patients(patient_id, token)
        if not patients:
            print(f"[Orthanc] No patient found with ID: {patient_id}")
            return []
        
        # Get the first matching patient
        orthanc_patient_id = patients[0].get('ID')
        if not orthanc_patient_id:
            print(f"[Orthanc] Invalid patient data: {patients[0]}")
            return []
            
        print(f"[Orthanc] Found patient with Orthanc ID: {orthanc_patient_id}")
        
        # Get patient information using Orthanc ID
        response = requests.get(
            f"{ORTHANC_SERVER_URL}/patients/{orthanc_patient_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            patient_data = response.json()
            print(f"[Orthanc] Found patient: {patient_data}")
            
            # Get all studies for this patient
            studies = patient_data.get('Studies', [])
            print(f"[Orthanc] Found {len(studies)} studies")
            
            instances = []
            for study_id in studies:
                print(f"[Orthanc] Getting instances for study: {study_id}")
                study_response = requests.get(
                    f"{ORTHANC_SERVER_URL}/studies/{study_id}/instances",
                    headers=headers
                )
                
                if study_response.status_code == 200:
                    study_instances = study_response.json()
                    print(f"[Orthanc] Found {len(study_instances)} instances in study")
                    instances.extend(study_instances)
                else:
                    print(f"[Orthanc] Error getting study {study_id}: {study_response.status_code}")
                    print(f"[Orthanc] Error response: {study_response.text}")
            
            print(f"[Orthanc] Total instances found: {len(instances)}")
            return instances
        else:
            print(f"[Orthanc] Error getting patient {patient_id}: {response.status_code}")
            print(f"[Orthanc] Error response: {response.text}")
            return []
            
    except Exception as e:
        print(f"[Orthanc] Error getting patient instances: {str(e)}")
        import traceback
        print(f"[Orthanc] Full error: {traceback.format_exc()}")
        return []

def get_instance_tags(instance_id, token):
    """
    Get DICOM tags of an instance.
    
    Args:
        instance_id (str): The Orthanc instance ID
        token (str): JWT token for authentication
        
    Returns:
        dict: Dictionary containing DICOM tags or None if not found
    """
    try:
        print(f"[Orthanc] Getting tags for instance: {instance_id}")
        response = requests.get(
            f"{ORTHANC_SERVER_URL}/instances/{instance_id}/simplified-tags",
            headers=get_headers(token)
        )
        
        if response.status_code == 200:
            tags = response.json()
            print(f"[Orthanc] Got tags: {tags}")
            return tags
        else:
            print(f"[Orthanc] Error getting tags: {response.status_code}")
            print(f"[Orthanc] Error response: {response.text}")
            return None
            
    except Exception as e:
        print(f"[Orthanc] Error getting tags: {str(e)}")
        import traceback
        print(f"[Orthanc] Full error: {traceback.format_exc()}")
        return None

def search_patients(query: str, token: str):
    """
    Search for patients in Orthanc PACS server by ID or name.
    
    Args:
        query (str): Search query for patient ID or name
        token (str): JWT token for authentication
        
    Returns:
        list: List of matching patient records
    """
    try:
        headers = get_headers(token)
        headers['Content-Type'] = 'application/json'
        
        # If no query parameters, get all patients
        if not query:
            response = requests.get(
                f"{ORTHANC_SERVER_URL}/patients?expand",
                headers=headers
            )
            if response.status_code == 200:
                return response.json()
            return []

        # If query parameters exist, get all patients first then filter in code
        response = requests.get(
            f"{ORTHANC_SERVER_URL}/patients?expand",
            headers=headers
        )
        
        if response.status_code != 200:
            return []
            
        all_patients = response.json()
        query = query.lower()
        
        # Perform fuzzy matching in code
        return [p for p in all_patients if 
                (p.get('MainDicomTags', {}).get('PatientID', '').lower().find(query) != -1) or
                (p.get('MainDicomTags', {}).get('PatientName', '').lower().find(query) != -1)]
        

            
    except Exception as e:
        logging.error(f"Error searching patients: {str(e)}")
        import traceback
        logging.error(f"Full error: {traceback.format_exc()}")
        return []

def get_instance_waveform(instance_id, token):
    """
    Get ECG waveform data from a DICOM instance.
    
    Args:
        instance_id (str): The Orthanc instance ID
        token (str): JWT token for authentication
        
    Returns:
        dict: Dictionary containing waveform data and metadata, or None if not found
        Format:
        {
            'waveform_data': list of waveform samples,
            'sampling_rate': sampling rate in Hz,
            'duration': duration in seconds,
            'leads': list of lead names,
            'units': measurement units
        }
    """
    # Get the DICOM file data
    logging.info(f"[get_instance_waveform] Getting DICOM data for instance: {instance_id}")
    dicom_data = get_instance_dicom(instance_id, token)
    if not dicom_data:
        logging.error(f"[get_instance_waveform] Failed to get DICOM data for instance: {instance_id}")
        return None
        
    try:
        # Load the DICOM data
        dataset = pydicom.dcmread(BytesIO(dicom_data))
        
        # Check if this is a waveform object
        if not hasattr(dataset, 'WaveformSequence'):
            logging.error("[get_instance_waveform] No waveform data found in DICOM file")
            return None
            
        # Get the first waveform sequence
        waveform = dataset.WaveformSequence[0]
        
        # Get the number of channels (leads)
        num_channels = waveform.NumberOfWaveformChannels
        
        # Get the number of samples
        num_samples = waveform.NumberOfWaveformSamples
        
        # Get the sampling rate
        sampling_rate = float(waveform.SamplingFrequency)
        
        # Get the waveform data
        waveform_data = waveform.WaveformData
        
        # Get channel definitions
        channel_definitions = waveform.ChannelDefinitionSequence
        
        # Extract lead names
        leads = []
        for channel in channel_definitions:
            leads.append(channel.ChannelSourceSequence[0].CodeMeaning)
            
        # Convert byte array to numpy array of 16-bit integers
        data = np.frombuffer(waveform_data, dtype=np.int16)
        
        # Reshape the data into channels
        data = data.reshape(num_samples, num_channels)
        
        # Calculate duration
        duration = num_samples / sampling_rate
        
        # Get units (usually mV)
        units = channel_definitions[0].ChannelSensitivity
        
        # Log waveform metadata without the actual waveform data
        logging.info(f"[get_instance_waveform] Successfully extracted waveform data:")
        logging.info(f"  - Number of leads: {num_channels}")
        logging.info(f"  - Number of samples: {num_samples}")
        logging.info(f"  - Sampling rate: {sampling_rate} Hz")
        logging.info(f"  - Duration: {duration} seconds")
        logging.info(f"  - Leads: {leads}")
        logging.info(f"  - Units: {units}")
        
        return {
            'waveform_data': data.tolist(),
            'sampling_rate': sampling_rate,
            'duration': duration,
            'leads': leads,
            'units': units
        }
        
    except Exception as e:
        logging.error(f"Error processing waveform data: {str(e)}")
        return None
