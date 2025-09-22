from typing import Optional, Dict, Any, List
import logging
import traceback
from datetime import datetime
from bson import ObjectId
from app.model.patient import Patient
from app.model.ecg_file import EcgFile

def interpret_diagnosis(diagnosis_result: Dict) -> Dict:
    """Interpret the diagnosis results from DNN and ResNet models.
    
    Args:
        diagnosis_result (dict): Raw diagnosis results from the models
        
    Returns:
        dict: Interpreted diagnosis information
    """
    try:
        # Handle empty or invalid diagnosis result
        if not diagnosis_result:
            return {
                'interpretation': 'Waiting for AI model analysis...',
                'heart_rate': 'Not Available',
                'rhythm': 'Not Available',
                'probabilities': {
                    'original_dnn_model': {},
                    'resnet_fine_tuned_model': {}
                }
            }
            
        # Get model results - handle both naming conventions
        dnn_results = diagnosis_result.get('original_dnn_model', diagnosis_result.get('original dnn model', {}))
        resnet_results = diagnosis_result.get('resnet_fine_tuned_model', diagnosis_result.get('resnet fine tuned model', {}))
        
        # Extract heart rate if directly available (unlikely in current format)
        heart_rate = diagnosis_result.get('heart_rate', 'Not Available')
        
        # Initialize rhythm as Not Available, we'll try to infer it from model results
        rhythm = 'Not Available'
        
        # Generate interpretation
        conditions = []
        
        # Determine rhythm from DNN model (Arrhythmia Classification)
        if dnn_results:
            # Define arrhythmia types and their full names
            arrhythmia_types = {
                '1dAVB': 'First-degree Atrioventricular Block',
                'RBBB': 'Right Bundle Branch Block',
                'LBBB': 'Left Bundle Branch Block',
                'SBRAD': 'Sinus Bradycardia',
                'AFIB': 'Atrial Fibrillation',
                'STACH': 'Sinus Tachycardia'
            }
            
            # Find the condition with highest probability
            try:
                # Convert all values to float for comparison
                float_results = {k: float(v) for k, v in dnn_results.items()}
                
                # Find the maximum probability
                if float_results:
                    max_key = max(float_results, key=float_results.get)
                    max_prob = float_results[max_key]
                    
                    # If highest probability is very low, likely normal rhythm
                    if max_prob < 0.01:
                        rhythm = "Normal Sinus Rhythm"
                        conditions.append(f"Rhythm Analysis: Normal Sinus Rhythm (inferred from low arrhythmia probabilities)")
                    else:
                        # Use the highest probability arrhythmia
                        condition = arrhythmia_types.get(max_key, max_key)
                        rhythm = condition
                        # Format with confidence percentage
                        conditions.append(f"Rhythm Analysis: {condition} (Confidence: {max_prob*100:.1f}%)")
                        # Example: "Rhythm Analysis: Atrial Fibrillation (Confidence: 7.4%)")
                else:
                    rhythm = "Not Available"
                    conditions.append("Rhythm Analysis: No data available")
            except (ValueError, TypeError) as e:
                logging.warning(f"Could not determine rhythm from DNN results: {dnn_results}, error: {str(e)}")
                rhythm = "Normal Sinus Rhythm (default)"
        
        # ResNet model interpretation (Morphology Classification)
        morphology_condition = None
        if resnet_results:
            try:
                # Convert all values to float for comparison
                float_results = {k: float(v) for k, v in resnet_results.items()}
                
                # Find the maximum probability
                if float_results:
                    max_key = max(float_results, key=float_results.get)
                    max_prob = float_results[max_key]
                    
                    # Define morphology types
                    morphology_types = {
                        'NORM': 'Normal',
                        'STTC': 'ST-T Changes',
                        'MI': 'Myocardial Infarction',
                        'HYP': 'Left Ventricular Hypertrophy',
                        'CD': 'Cardiac Disease',
                        'RBBB': 'Right Bundle Branch Block',
                        '1dAVB': 'First-degree AV Block',
                        '2dAVB': 'Second-degree AV Block',
                        '3dAVB': 'Third-degree AV Block'
                    }
                    
                    # Only add morphology analysis if there's a significant probability
                    if max_prob > 0.01:  # Only show if probability > 1%
                        # Get the condition name
                        condition = morphology_types.get(max_key, max_key)
                        morphology_condition = condition
                        
                        # Format the probability display
                        prob_display = f"{max_prob*100:.1f}%"
                        
                        conditions.append(f"Morphology Analysis: {condition} (Confidence: {prob_display})")
                        
                        # If it's NORM with high probability and rhythm is still not available
                        if max_key == 'NORM' and max_prob > 0.4 and rhythm == 'Not Available':
                            rhythm = "Normal Sinus Rhythm"
                else:
                    conditions.append("Morphology Analysis: No significant findings")
                
            except (ValueError, TypeError) as e:
                logging.warning(f"Could not determine morphology from ResNet results: {resnet_results}, error: {str(e)}")
        
        # Generate final interpretation
        if not conditions:
            interpretation = "No significant abnormalities detected"
        else:
            interpretation = ", ".join(conditions)
        
        # If we have a normal morphology but no specific rhythm, use a generic normal description
        if morphology_condition == "Normal" and rhythm == "Not Available":
            rhythm = "Normal Sinus Rhythm"
        
        return {
            'interpretation': interpretation,
            'heart_rate': heart_rate,
            'rhythm': rhythm,
            'probabilities': {
                'original_dnn_model': dnn_results,
                'resnet_fine_tuned_model': resnet_results
            }
        }
        
    except Exception as e:
        logging.error(f"Error interpreting diagnosis: {str(e)}")
        return {
            'interpretation': f"Diagnosis interpretation failed: {str(e)}",
            'heart_rate': 'Not Available',
            'rhythm': 'Not Available',
            'probabilities': {}
        }

# 3. Get ECG data from Orthanc
from app.utils.orthanc import get_instance_waveform
from app.utils.auth_utils import get_temp_token

def get_ecg_data(patient_id: Optional[str] = None, instance_id: Optional[str] = None, study_date: Optional[str] = None) -> Dict[str, Any]:
    """Get ECG data by either patient_id or instance_id.
    
    Args:
        patient_id (str, optional): Patient ID to get the latest ECG for
        instance_id (str, optional): The DICOM UID / instance ID to get a specific ECG
        study_date (str, optional): Study date in YYYYMMDD format (only used with patient_id)
        
    Returns:
        dict: Dictionary containing:
        - patient: Patient information
        - ecg_data: ECG data including waveform and metadata
        - error: Error message if any
        
    Note: If both patient_id and instance_id are provided, instance_id takes precedence.
    """
    if instance_id:
        return get_ecg_by_instance_id(instance_id)
    elif patient_id:
        return get_patient_ecg_data(patient_id, study_date)
    else:
        return {"error": "Either patient_id or instance_id must be provided"}

def get_ecg_by_instance_id(instance_id: str) -> Dict[str, Any]:
    """Get ECG data for a specific instance ID (dicom_uid).
    
    Args:
        instance_id (str): The DICOM UID / instance ID
        
    Returns:
        dict: Dictionary containing:
        - patient: Patient information
        - ecg_data: ECG data including waveform and metadata
        - error: Error message if any
    """
    try:
        # Find the ECG file by dicom_uid
        try:
            logging.info(f"[get_ecg_by_instance_id] Looking for ECG with instance ID: {instance_id}")
            ecg_file = EcgFile.objects(dicom_uid=instance_id).first()
            if not ecg_file:
                return {"error": f"ECG not found with instance ID: {instance_id}"}
            logging.info(f"[get_ecg_by_instance_id] Found ECG file: {ecg_file.dicom_uid}, Created At: {ecg_file.created_at}")
        except Exception as e:
            return {"error": f"Error getting ECG: {str(e)}"}
        
        # Get patient info
        try:
            # Since patient_id is a ReferenceField to Patient in the EcgFile model,
            # ecg_file.patient_id should already be a Patient object
            logging.info(f"[get_ecg_by_instance_id] Patient ID type: {type(ecg_file.patient_id)}")
            
            # Just use the patient_id directly as it's already a Patient object
            patient = ecg_file.patient_id
            
            # Verify that we have a valid Patient object
            if not patient or not isinstance(patient, Patient):
                logging.warning(f"[get_ecg_by_instance_id] Invalid patient reference: {patient}")
                # If not a valid Patient object, try to get by ID
                try:
                    if hasattr(patient, 'id'):
                        patient = Patient.objects(id=patient.id).first()
                    else:
                        patient = None
                except Exception as e:
                    logging.error(f"[get_ecg_by_instance_id] Error getting patient by ID: {str(e)}")
                    patient = None
                
            if not patient:
                return {"error": f"Patient not found for ECG instance: {instance_id}"}
            logging.info(f"[get_ecg_by_instance_id] Found patient: {patient.id} ({patient.name})")
        except Exception as e:
            return {"error": f"Error getting patient: {str(e)}"}
        
        # Get temporary token for Orthanc API access
        try:
            logging.info(f"[get_ecg_by_instance_id] Getting temp token for Orthanc access")
            token = get_temp_token()
            if not token:
                logging.error(f"[get_ecg_by_instance_id] Failed to get auth token")
                return {"error": "Failed to get authentication token"}
            logging.info(f"[get_ecg_by_instance_id] Got token, length: {len(token)}")
            
            logging.info(f"[get_ecg_by_instance_id] Getting waveform data for DICOM: {ecg_file.dicom_uid}")
            waveform_data = get_instance_waveform(ecg_file.dicom_uid, token)
            logging.info(f"[get_ecg_by_instance_id] Waveform data retrieved: {bool(waveform_data)}")
        except Exception as e:
            logging.error(f"[get_ecg_by_instance_id] Error getting waveform data: {str(e)}")
            logging.error(traceback.format_exc())
            return {"error": f"Error accessing ECG data: {str(e)}"}
        
        # Get diagnosis interpretation
        if not ecg_file.diagnosis_result:
            logging.warning(f"[get_ecg_by_instance_id] No diagnosis result found for ECG file: {ecg_file.dicom_uid}")
            
        diagnosis = interpret_diagnosis(ecg_file.diagnosis_result)
        # Log diagnostic interpretation results and original data status
        logging.info(f"[get_ecg_by_instance_id] Raw diagnosis data exists: {bool(ecg_file.diagnosis_result)}")
        logging.info(f"[get_ecg_by_instance_id] Diagnosis interpretation: {diagnosis.get('interpretation', 'N/A')}")
        
        if not waveform_data:
            return {"error": f"Failed to get ECG waveform data from Orthanc server"}
            
        # Format ECG data
        study_datetime = ecg_file.created_at
        ecg_data = {
            "study_date": study_datetime.strftime("%Y%m%d"),
            "study_time": study_datetime.strftime("%H%M%S"),
            "dicom_uid": ecg_file.dicom_uid,  # Include dicom_uid as instance ID
            "waveform": waveform_data,
            "diagnosis": {
                "raw_results": ecg_file.diagnosis_result,
                "interpretation": diagnosis["interpretation"],
                "probabilities": diagnosis["probabilities"]
            }
        }
        
        # Format patient data
        patient_data = {
            "patient_id": patient.patient_id,
            "name": patient.name,
            "gender": patient.gender,
            "birth_date": patient.birth_date.strftime("%Y-%m-%d") if patient.birth_date else None
        }
        
        return {
            "patient": patient_data,
            "ecg_data": ecg_data
        }
        
    except Exception as e:
        logging.error(f"[get_ecg_by_instance_id] Error getting ECG data: {str(e)}")
        return {"error": f"Failed to get ECG data: {str(e)}"}

def get_patient_ecg_data(patient_id: str, study_date: Optional[str] = None) -> Dict[str, Any]:
    """Get ECG data for a patient from database and Orthanc server.
    
    Args:
        patient_id (str): Patient ID
        study_date (str, optional): Study date in YYYYMMDD format
        
    Returns:
        dict: Dictionary containing:
        - patient: Patient information
        - ecg_data: ECG data including waveform and metadata
        - error: Error message if any
    """
    try:
        # Get patient info
        try:
            logging.info(f"[get_patient_ecg_data] Looking for patient with ID: {patient_id}")
            patient = Patient.objects(patient_id=patient_id).first()
            if patient:
                logging.info(f"[get_patient_ecg_data] Found patient: {patient.id} ({patient.name})")
            if not patient:
                return {"error": f"Patient not found with ID: {patient_id}"}
        except Exception as e:
            return {"error": f"Error getting patient: {str(e)}"}
        
        # Get ECG data
        try:
            logging.info(f"[get_patient_ecg_data] Patient ID type: {type(patient.id)}")
            if isinstance(patient.id, str):
                patient_id_obj = ObjectId(patient.id)
            elif isinstance(patient.id, ObjectId):
                patient_id_obj = patient.id
            else:
                patient_id_obj = ObjectId(str(patient.id))
            query = {"patient_id": patient_id_obj}
            logging.info(f"[get_patient_ecg_data] Query: {query}")
            if study_date:
                # Convert study_date string to datetime range
                from datetime import datetime
                start_date = datetime.strptime(study_date, "%Y%m%d")
                end_date = datetime.strptime(f"{study_date}235959", "%Y%m%d%H%M%S")
                query["created_at__gte"] = start_date
                query["created_at__lte"] = end_date
            ecg_file = EcgFile.objects(**query).order_by("-created_at").first()
            if not ecg_file:
                return {"error": f"No ECG records found for patient: {patient.name}"}
            
            # Log minimal ECG file details
            logging.info(f"[get_patient_ecg_data] Found ECG file: {ecg_file.dicom_uid}, Created At: {ecg_file.created_at},diagnosis_result: {ecg_file.diagnosis_result}")
        except Exception as e:
            return {"error": f"Error getting ECG data: {str(e)}"}
        
        # Get temporary token for Orthanc API access
        try:
            logging.info(f"[get_patient_ecg_data] Getting temp token for Orthanc access")
            token = get_temp_token()
            if not token:
                logging.error(f"[get_patient_ecg_data] Failed to get auth token")
                return {"error": "Failed to get authentication token"}
            logging.info(f"[get_patient_ecg_data] Got token, length: {len(token)}")
            
            logging.info(f"[get_patient_ecg_data] Getting waveform data for DICOM: {ecg_file.dicom_uid}")
            waveform_data = get_instance_waveform(ecg_file.dicom_uid, token)
            logging.info(f"[get_patient_ecg_data] Waveform data retrieved: {bool(waveform_data)}")
        except Exception as e:
            logging.error(f"[get_patient_ecg_data] Error getting waveform data: {str(e)}")
            logging.error(traceback.format_exc())
            return {"error": f"Error accessing ECG data: {str(e)}"}
        
        # Get diagnosis interpretation
        if not ecg_file.diagnosis_result:
            logging.warning(f"[get_patient_ecg_data] No diagnosis result found for ECG file: {ecg_file.dicom_uid}")
            
        diagnosis = interpret_diagnosis(ecg_file.diagnosis_result)
        # Log diagnostic interpretation results and original data status
        logging.info(f"[get_patient_ecg_data] Raw diagnosis data exists: {bool(ecg_file.diagnosis_result)}")
        logging.info(f"[get_patient_ecg_data] Diagnosis interpretation: {diagnosis.get('interpretation', 'N/A')}")
        
        if not waveform_data:
            return {"error": f"Failed to get ECG waveform data from Orthanc server"}
            
        # Format ECG data
        study_datetime = ecg_file.created_at
        ecg_data = {
            "study_date": study_datetime.strftime("%Y%m%d"),
            "study_time": study_datetime.strftime("%H%M%S"),
            "dicom_uid": ecg_file.dicom_uid,  # Include dicom_uid as instance ID
            "waveform": waveform_data,
            "diagnosis": {
                "raw_results": ecg_file.diagnosis_result,
                "interpretation": diagnosis["interpretation"],
                "probabilities": diagnosis["probabilities"]
            }
        }
        
        # Format patient data
        patient_data = {
            "patient_id": patient.patient_id,
            "name": patient.name,
            "gender": patient.gender,
            "birth_date": patient.birth_date.strftime("%Y-%m-%d") if patient.birth_date else None
        }
        
        return {
            "patient": patient_data,
            "ecg_data": ecg_data
        }
        
    except Exception as e:
        logging.error(f"Error getting ECG data: {str(e)}")
        return {"error": f"Failed to get ECG data: {str(e)}"}
