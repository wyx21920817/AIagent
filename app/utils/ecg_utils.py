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
                    'original dnn model': {},
                    'resnet fine tuned model': {}
                }
            }
            
        # Get model results
        dnn_results = diagnosis_result.get('original dnn model', {})
        resnet_results = diagnosis_result.get('resnet fine tuned model', {})
        
        # Generate interpretation
        conditions = []
        
        # DNN model interpretation (Arrhythmia Classification)
        if dnn_results:
            dnn_max = max(dnn_results.items(), key=lambda x: float(x[1]))
            prob = float(dnn_max[1])
            if prob > 0.5:  # Only include if probability > 50%
                arrhythmia_types = {
                    '1dAVb': 'First-degree Atrioventricular Block',
                    'RBBB': 'Right Bundle Branch Block',
                    'LBBB': 'Left Bundle Branch Block',
                    'SB': 'Sinus Bradycardia',
                    'AF': 'Atrial Fibrillation',
                    'ST': 'Sinus Tachycardia'
                }
                condition = arrhythmia_types.get(dnn_max[0], dnn_max[0])
                conditions.append(f"Rhythm Analysis: {condition} (Confidence: {prob:.1%})")
        
        # ResNet model interpretation (Morphology Classification)
        if resnet_results:
            resnet_max = max(resnet_results.items(), key=lambda x: float(x[1]))
            prob = float(resnet_max[1]) / 100.0  # ResNet probabilities are 0-100
            if prob > 0.5:  # Only include if probability > 50%
                morphology_types = {
                    'NORM': 'Normal',
                    'STTC': 'ST-T Changes',
                    'MI': 'Myocardial Infarction',
                    'HYP': 'Left Ventricular Hypertrophy',
                    'CD': 'Cardiac Disease'
                }
                condition = morphology_types.get(resnet_max[0], resnet_max[0])
                conditions.append(f"Morphology Analysis: {condition} (Confidence: {prob:.1%})")
            
        # Generate final interpretation
        if not conditions:
            interpretation = "No significant abnormalities detected"
        else:
            interpretation = ", ".join(conditions)
        
        return {
            'interpretation': interpretation,
            'probabilities': {
                'original dnn model': dnn_results,
                'resnet fine tuned model': resnet_results
            }
        }
        
    except Exception as e:
        logging.error(f"Error interpreting diagnosis: {str(e)}")
        return {
            'interpretation': f"Diagnosis interpretation failed: {str(e)}",
            'probabilities': {}
        }

# 3. Get ECG data from Orthanc
from app.utils.orthanc import get_instance_waveform
from app.utils.auth_utils import get_temp_token

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
