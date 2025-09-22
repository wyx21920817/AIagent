from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import logging
import os
from datetime import datetime
from enum import Enum, auto
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from app.utils.models import ChatResponse, SessionData
from app.core.database import Database
from app.model.ecg_file import EcgFile
from app.model.patient import Patient
from app.data.memory.session import SessionManager
from app.utils.ecg_utils import get_patient_ecg_data, interpret_diagnosis
from app.utils.ecg_plot import plot_ecg_waveform
from app.utils.pdf_utils import generate_pdf_report, get_report_download_url
from app.utils.auth_utils import get_temp_token
import numpy as np
import os

# Initialize managers
session_manager = SessionManager()
db = Database()

class Intent(str, Enum):
    """Intent types for the medical assistant"""
    QUERY_PATIENT = "query_patient"         # Search for or query patient information
    SWITCH_PATIENT = "switch_patient"       # Switch current patient context
    VIEW_ECG = "view_ecg"                  # View ECG waveform data
    ANALYZE_ECG = "analyze_ecg"            # Analyze ECG or view analysis results
    GENERATE_REPORT = "generate_report"    # Generate diagnostic report
    MEDICAL_QUESTION = "medical_question"  # General medical questions
    EXPORT_REPORT = "export_report"        # Export report
    COMPARE_PATIENTS = "compare_patients"  # Compare patient data
    OTHER = "other"                       # Other intents

class IntentResult:
    def __init__(self, intent: Intent, confidence: float, extracted_info: Dict[str, Any] = None):
        self.intent = intent
        self.confidence = confidence
        self.extracted_info = extracted_info or {}

async def get_latest_diagnosis(db: Database, study_date: Optional[str] = None, instance_id: Optional[str] = None) -> Optional[Dict]:
    """Get diagnosis by instance_id or study_date"""
    try:
        # If instance_id is provided, try to get the ECG file directly by dicom_uid
        if instance_id:
            from app.model.ecg_file import EcgFile
            try:
                ecg_file = EcgFile.objects(dicom_uid=instance_id).first()
                if ecg_file and hasattr(ecg_file, 'diagnosis_result') and ecg_file.diagnosis_result:
                    diagnosis = ecg_file.diagnosis_result
                    # Ensure diagnosis result includes timestamp
                    if "diagnosis_time" not in diagnosis:
                        diagnosis["diagnosis_time"] = ecg_file.created_at
                    if "created_at" not in diagnosis:
                        diagnosis["created_at"] = ecg_file.created_at
                        
                    # Add some metadata
                    diagnosis["ecg_id"] = str(ecg_file.id)
                    
                    logging.info(f"[get_latest_diagnosis] Found diagnosis for instance {instance_id}")
                    return diagnosis
                else:
                    logging.info(f"[get_latest_diagnosis] No diagnosis found for instance {instance_id}")
            except Exception as e:
                logging.error(f"[get_latest_diagnosis] Error finding ECG file by dicom_uid: {str(e)}", exc_info=True)
        
        # If no instance_id or no diagnosis found by instance_id, we can't proceed without patient_id
        logging.warning(f"[get_latest_diagnosis] Cannot get diagnosis without instance_id")
        return None
        
    except Exception as e:
        logging.error(f"[get_latest_diagnosis] Failed to get diagnosis: {str(e)}", exc_info=True)
        return None


async def handle_switch_patient(extracted_info: Dict[str, Any], conversation_id: str) -> ChatResponse:
    """Switch to another patient and show their key information"""
    logging.info(f"[handle_switch_patient] Processing switch patient intent with info: {extracted_info}")
    response = await handle_patient_query(extracted_info, conversation_id)
    
    # Add a note about switching patient if successful
    if response and response.patient_context and 'patient_id' in response.patient_context:
        response.message = "Switched to patient:\n" + response.message
    
    return response

async def handle_patient_query(extracted_info: Dict[str, Any], conversation_id: str) -> ChatResponse:
    """Handle patient information query intent"""
    logging.info(f"[handle_patient_query] Processing query with extracted_info: {extracted_info}")
    patient = None
    
    # Get query parameters
    patient_name = extracted_info.get("patient_name")
    patient_id = extracted_info.get("patient_id")
    
    # Try ID search first, as it's more precise
    if patient_id:
        try:
            logging.info(f"[handle_patient_query] Searching for patient by ID: {patient_id}")
            patient = await db.get_patient(patient_id)
        except Exception as e:
            logging.error(f"[handle_patient_query] Error finding patient by ID: {str(e)}")
            return ChatResponse(
                message=f"Error finding patient: {str(e)}",
                conversation_id=conversation_id
            )
    
    # If not found, try name search
    if not patient and patient_name:
        try:
            logging.info(f"[handle_patient_query] Searching for patient by name: {patient_name}")
            patient = await db.find_patient_by_name(patient_name)
        except Exception as e:
            logging.error(f"[handle_patient_query] Error finding patient by name: {str(e)}")
            return ChatResponse(
                message=f"Error finding patient: {str(e)}",
                conversation_id=conversation_id
            )
    
    if not patient:
        logging.info("[handle_patient_query] Patient not found")
        return ChatResponse(
            message="No patient found with the given information.",
            conversation_id=conversation_id
        )
    
    # Get basic patient information
    info = [
        f"Patient Name: {patient.name}",
        f"Patient ID: {patient.patient_id}",
        f"Gender: {patient.gender if patient.gender else 'Not specified'}",
        f"Birth Date: {patient.birth_date.strftime('%Y-%m-%d') if patient.birth_date else 'Not specified'}"
    ]
    
    # We can't get the latest diagnosis without an instance_id anymore
    # Try to get the instance_id from the session if available
    diagnosis = None
    try:
        session = await session_manager.get_session(conversation_id)
        instance_id = None
        if session and hasattr(session, 'ecg_instance_id') and session.ecg_instance_id:
            instance_id = session.ecg_instance_id
            logging.info(f"[handle_patient_query] Found instance_id {instance_id} in session")
            
            # Get diagnosis using the instance_id
            diagnosis = await get_latest_diagnosis(db, instance_id=instance_id)
            if diagnosis:
                # Format diagnostic results display
                info.extend([
                    "",
                    "Latest Diagnosis:",
                    f"Date: {diagnosis.get('diagnosis_time', 'Unknown')}"
                ])
            else:
                logging.info(f"[handle_patient_query] No diagnosis found for instance {instance_id}")
        else:
            logging.info(f"[handle_patient_query] No instance_id found in session")
    except Exception as e:
        logging.error(f"[handle_patient_query] Error getting diagnosis: {str(e)}", exc_info=True)
            
    # Format DNN model results if diagnosis exists
    if diagnosis:
        # Format DNN model results
        dnn_probs = diagnosis.get('original_dnn_model', {})
        if dnn_probs:
            info.append("\nArrhythmia Analysis (DNN Model):")
            for condition, prob in sorted(dnn_probs.items(), key=lambda x: float(x[1]), reverse=True):
                if float(prob) > 0.01:  # Only show results with probability > 1%
                    info.append(f"- {condition}: {float(prob)*100:.1f}%")
        
        # Format ResNet model results
        resnet_probs = diagnosis.get('resnet_fine_tuned_model', {})
        if resnet_probs:
            info.append("\nMorphological Analysis (ResNet Model):")
            for condition, prob in sorted(resnet_probs.items(), key=lambda x: float(x[1]), reverse=True):
                if float(prob) > 0.01:  # Only show results with probability > 1%
                    info.append(f"- {condition}: {float(prob)*100:.1f}%")
    
    # Update the session with the new patient ID and clear the ecg_instance_id
    try:
        session = await session_manager.get_session(conversation_id)
        if session:
            # Check if patient ID has changed
            if session.patient_id != patient.patient_id:
                # Clear the ecg_instance_id when switching to a new patient
                session.ecg_instance_id = None
                logging.info(f"[handle_patient_query] Cleared ecg_instance_id due to patient change from {session.patient_id} to {patient.patient_id}")
            
            # Update the patient ID
            session.patient_id = patient.patient_id
            await session_manager.save_session(session)
            logging.info(f"[handle_patient_query] Updated session with patient_id {patient.patient_id}")
    except Exception as e:
        logging.error(f"[handle_patient_query] Error updating session: {str(e)}", exc_info=True)
    
    response = ChatResponse(
        message="Patient Information:\n" + "\n".join(f"    {line}" for line in info),
        conversation_id=conversation_id,
        patient_context={
            "patient_id": patient.patient_id,
            "name": patient.name
        }
    )
    
    # Log complete response
    logging.info(f"[handle_patient_query] Returning response:\n" + 
                f"Message: {response.message}\n" + 
                f"Conversation ID: {response.conversation_id}\n" + 
                f"Patient Context: {response.patient_context}")
    
    return response



async def handle_analyze_ecg(db, extracted_info: Dict[str, Any], conversation_id: str) -> ChatResponse:
    """Handle request to analyze ECG data"""
    logging.info(f"[handle_analyze_ecg] Starting ECG analysis with extracted_info: {extracted_info}")
    
    logging.info(f"[handle_analyze_ecg] Using conversation_id: {conversation_id}")
    
    # Get patient ID, prioritize from session
    session = await session_manager.get_session(conversation_id)
    patient_id = None
    
    # 1. First check if patient_id exists in extracted_info
    if "patient_id" in extracted_info and extracted_info["patient_id"]:
        patient_id = extracted_info["patient_id"]
    # 2. If not, check if patient_id exists in session
    elif session and session.patient_id:
        patient_id = session.patient_id
        logging.info(f"[handle_analyze_ecg] Using patient_id from session: {patient_id}")
    else:
        logging.warning("[handle_analyze_ecg] No patient_id in extracted_info or session")
        return ChatResponse(
            message="Please specify a patient ID to analyze ECG data.",
            conversation_id=conversation_id
        )
        
    logging.info(f"[handle_analyze_ecg] Analyzing ECG for patient: {patient_id}")
    
    # Study date is no longer in the prompt
    study_date = None
    
    # Get ECG data using the utility function
    from app.utils.ecg_utils import get_patient_ecg_data
    result = get_patient_ecg_data(patient_id, study_date)
    
    if "error" in result:
        logging.warning(f"[handle_analyze_ecg] {result['error']} for patient {patient_id}")
        return ChatResponse(
            message=result["error"],
            conversation_id=conversation_id
        )
        
    # Store the ECG instance ID (dicom_uid) in the session
    try:
        # Get the dicom_uid from the result
        ecg_data = result.get("ecg_data", {})
        dicom_uid = ecg_data.get("dicom_uid")
        
        if dicom_uid:
            # Update the session with the ECG instance ID
            session = await session_manager.get_session(conversation_id)
            if session:
                session.ecg_instance_id = dicom_uid
                await session_manager.save_session(session)
                logging.info(f"[handle_analyze_ecg] Stored ecg_instance_id {dicom_uid} in session for patient {patient_id}")
    except Exception as e:
        logging.error(f"[handle_analyze_ecg] Error updating session with ecg_instance_id: {str(e)}", exc_info=True)
    
    # Get ECG data
    ecg_data = result["ecg_data"]
    
    # Get the diagnosis using the instance_id (dicom_uid)
    dicom_uid = ecg_data.get("dicom_uid")
    if not dicom_uid:
        logging.warning(f"[handle_analyze_ecg] No dicom_uid found in ECG data")
        return ChatResponse(
            message="No ECG instance ID found for analysis.",
            conversation_id=conversation_id
        )
        
    diagnosis = await get_latest_diagnosis(db, instance_id=dicom_uid)
    if not diagnosis:
        logging.warning(f"[handle_analyze_ecg] No diagnosis found for instance {dicom_uid}")
        return ChatResponse(
            message="No diagnosis found for this ECG instance.",
            conversation_id=conversation_id
        )
    
    # Log the diagnosis for debugging
    logging.info(f"[handle_analyze_ecg] Diagnosis: {diagnosis}")
    
    # Format diagnosis probabilities directly from the diagnosis
    prob_lines = []
    
    # DNN model probabilities (Arrhythmia Analysis)
    dnn_probs = diagnosis.get('original_dnn_model', {})
    if dnn_probs:
        arrhythmia_types = {
            '1dAVB': 'First-degree AV block',
            'RBBB': 'Right Bundle Branch Block',
            'LBBB': 'Left Bundle Branch Block',
            'SBRAD': 'Sinus Bradycardia',
            'AFIB': 'Atrial Fibrillation',
            'STACH': 'Sinus Tachycardia'
        }
        prob_lines.append("\nArrhythmia Analysis:")
        for condition, prob in dnn_probs.items():
            if isinstance(prob, (int, float)) and float(prob) > 0.01:  # Only show if probability > 1%
                condition_name = arrhythmia_types.get(condition, condition)
                prob_lines.append(f"- {condition_name}: {float(prob)*100:.1f}%")
    
    # ResNet model probabilities (Morphological Analysis)
    resnet_probs = diagnosis.get('resnet_fine_tuned_model', {})
    if resnet_probs:
        morphology_types = {
            'NORM': 'Normal',
            'STTC': 'ST-T Changes',
            'MI': 'Myocardial Infarction',
            'HYP': 'Left Ventricular Hypertrophy',
            'CD': 'Cardiac Disease'
        }
        prob_lines.append("\nMorphological Analysis:")
        for condition, prob in resnet_probs.items():
            if isinstance(prob, (int, float)) and float(prob) > 1.0:  # ResNet probabilities are 0-100
                condition_name = morphology_types.get(condition, condition)
                prob_lines.append(f"- {condition_name}: {float(prob):.1f}%")
    
    logging.info(f"[handle_analyze_ecg] Analyzed ECG for patient {patient_id} with diagnosis probabilities")
    
    # Format diagnostic results
    formatted_lines = ["Diagnostic Results:"]
    
    # Format DNN model results
    dnn_probs = diagnosis.get('original_dnn_model', {})
    if dnn_probs:
        formatted_lines.append("\nArrhythmia Analysis (DNN Model):")
        for condition, prob in sorted(dnn_probs.items(), key=lambda x: float(x[1]), reverse=True):
            if float(prob) > 0.01:  # Only show results with probability > 1%
                formatted_lines.append(f"- {condition}: {float(prob)*100:.1f}%")
    
    # Format ResNet model results
    resnet_probs = diagnosis.get('resnet_fine_tuned_model', {})
    if resnet_probs:
        formatted_lines.append("\nMorphological Analysis (ResNet Model):")
        for condition, prob in sorted(resnet_probs.items(), key=lambda x: float(x[1]), reverse=True):
            if float(prob) > 0.01:  # Only show results with probability > 1%
                formatted_lines.append(f"- {condition}: {float(prob)*100:.1f}%")
    
    return ChatResponse(
        message=f"ECG Analysis for {result['patient']['name']}:\n\n" + 
                "\n".join(formatted_lines),
        conversation_id=conversation_id,
        metadata={
            "diagnosis": diagnosis,
            "patient_context": {
                "patient_id": patient_id,
                "patient_name": result['patient']['name'],
                "study_date": ecg_data['study_date'],
                "study_time": ecg_data['study_time']
            }
        }
    )

async def handle_view_ecg(db, extracted_info: Dict[str, Any], conversation_id: str) -> ChatResponse:
    """Handle request to view ECG data"""
    # Filter out image data from logging
    log_info = {k:v for k,v in extracted_info.items() if k != 'waveform_data'}
    logging.info(f"[handle_view_ecg] Starting ECG view with extracted_info: {log_info}")
    
    # Get patient ID, prioritize from session
    session = await session_manager.get_session(conversation_id)
    patient_id = None
    
    # Check if instance_id/ecg_id is specified in the request
    instance_id = extracted_info.get("instance_id") or extracted_info.get("ecg_id") or extracted_info.get("report_id")
    if instance_id:
        logging.info(f"[handle_view_ecg] Instance ID specified in request: {instance_id}")
        # Update the session with the requested instance_id immediately
        try:
            if session:
                session.ecg_instance_id = instance_id
                await session_manager.save_session(session)
                logging.info(f"[handle_view_ecg] Updated session with requested instance_id: {instance_id}")
        except Exception as e:
            logging.error(f"[handle_view_ecg] Error updating session with requested instance_id: {str(e)}", exc_info=True)
    
    # 1. First check if patient_id exists in extracted_info
    if "patient_id" in extracted_info and extracted_info["patient_id"]:
        patient_id = extracted_info["patient_id"]
        logging.info(f"[handle_view_ecg] Using patient_id from extracted_info: {patient_id}")
    
    # 2. If not, check if patient_id exists in session
    elif session and session.patient_id:
        patient_id = session.patient_id
        logging.info(f"[handle_view_ecg] Using patient_id from session: {patient_id}")
        # Update extracted_info for consistency
        extracted_info["patient_id"] = patient_id
    
    if not patient_id:
        return ChatResponse(
            message="Please select a patient first to view ECG data.",
            conversation_id=conversation_id
        )
    
    # Study date is no longer in the prompt
    study_date = None
    
    # Get ECG data using the utility function
    from app.utils.ecg_utils import get_patient_ecg_data
    result = get_patient_ecg_data(patient_id, study_date)
    
    if "error" in result:
        return ChatResponse(
            message=result["error"],
            conversation_id=conversation_id
        )
    
    # Format date and time
    ecg_data = result["ecg_data"]
    date_str = f"{ecg_data['study_date'][:4]}-{ecg_data['study_date'][4:6]}-{ecg_data['study_date'][6:]}"
    time_str = f"{ecg_data['study_time'][:2]}:{ecg_data['study_time'][2:4]}"
    
    # Store the ECG instance ID (dicom_uid) in the session
    try:
        # Get the dicom_uid from the result
        dicom_uid = ecg_data.get("dicom_uid")
        
        if dicom_uid:
            # Update the session with the ECG instance ID
            session = await session_manager.get_session(conversation_id)
            if session:
                session.ecg_instance_id = dicom_uid
                await session_manager.save_session(session)
                logging.info(f"[handle_view_ecg] Stored ecg_instance_id {dicom_uid} in session for patient {patient_id}")
    except Exception as e:
        logging.error(f"[handle_view_ecg] Error updating session with ecg_instance_id: {str(e)}", exc_info=True)
    
    # Prepare response message
    info_lines = [
        f"ECG Record for {result['patient']['name']}",
        "=======================\n",
        f"Date: {date_str}",
        f"Time: {time_str}"
    ]
    
    # Add waveform info and plot
    waveform = ecg_data.get("waveform", {})
    if waveform:
        info_lines.extend([
            "",
            "Waveform Information:",
            f"- Sampling Rate: {waveform.get('sampling_rate', 'Not Available')} Hz",
            f"- Duration: {waveform.get('duration', 'Not Available')} seconds",
            f"- Number of Leads: {len(waveform.get('leads', []))}",
            f"- Leads: {', '.join(waveform.get('leads', ['Not Available']))}",
            f"- Units: {waveform.get('units', 'Not Available')}"
        ])
        
        # Generate and add ECG plot
        try:
            waveform_data = np.array(waveform['waveform_data'])
            plot_base64 = plot_ecg_waveform(
                waveform_data,
                waveform['sampling_rate'],
                waveform['leads'],
                waveform['duration']
            )
            info_lines.extend([
                "",
                "ECG Waveform Plot:",
                f"![ECG Waveform](data:image/png;base64,{plot_base64})"
            ])
        except Exception as e:
            logging.error(f"Error generating ECG plot: {str(e)}")
            info_lines.extend([
                "",
                "Error: Could not generate ECG waveform plot."
            ])
    
    # Get the diagnosis using the instance_id (dicom_uid)
    dicom_uid = ecg_data.get("dicom_uid")
    diagnosis = None
    if dicom_uid:
        diagnosis = await get_latest_diagnosis(db, instance_id=dicom_uid)
        if diagnosis:
            logging.info(f"[handle_view_ecg] Found diagnosis for instance {dicom_uid}")
        else:
            logging.warning(f"[handle_view_ecg] No diagnosis found for instance {dicom_uid}")
    else:
        logging.warning(f"[handle_view_ecg] No dicom_uid found in ECG data")
    
    if diagnosis:
        # Format diagnostic results display
        info_lines.extend([
            "",
            "Diagnostic Results:"
        ])
        
        # Format DNN model results
        dnn_probs = diagnosis.get('original_dnn_model', {})
        if dnn_probs:
            info_lines.append("\nArrhythmia Analysis (DNN Model):")
            for condition, prob in sorted(dnn_probs.items(), key=lambda x: float(x[1]), reverse=True):
                if float(prob) > 0.01:  # Only show results with probability > 1%
                    info_lines.append(f"- {condition}: {float(prob)*100:.1f}%")
        
        # Format ResNet model results
        resnet_probs = diagnosis.get('resnet_fine_tuned_model', {})
        if resnet_probs:
            info_lines.append("\nMorphological Analysis (ResNet Model):")
            for condition, prob in sorted(resnet_probs.items(), key=lambda x: float(x[1]), reverse=True):
                if float(prob) > 0.01:  # Only show results with probability > 1%
                    info_lines.append(f"- {condition}: {float(prob)*100:.1f}%")
        
        # We don't need to add model analysis results anymore since we're showing the raw diagnosis_result
    
    return ChatResponse(
        message="\n".join(info_lines),
        conversation_id=conversation_id,
        patient_context={
            "patient_id": patient_id,
            "ecg_metadata": {
                "sampling_rate": ecg_data["waveform"].get("sampling_rate"),
                "duration": ecg_data["waveform"].get("duration"),
                "leads": ecg_data["waveform"].get("leads"),
                "units": ecg_data["waveform"].get("units")
            },
            "action": "display_waveform"
        }
    )

async def handle_generate_report(db, extracted_info: Dict[str, Any], conversation_id: str) -> ChatResponse:
    """Generate a diagnostic report"""
    patient_id = extracted_info.get("patient_id")
    if not patient_id:
        return ChatResponse(
            message="Please specify a patient ID to generate report.",
            conversation_id=conversation_id
        )
    
    # Study date is no longer in the prompt
    study_date = None
    
    # Get ECG data using the utility function
    from app.utils.ecg_utils import get_patient_ecg_data
    result = get_patient_ecg_data(patient_id, study_date)
    
    if "error" in result:
        return ChatResponse(
            message=result["error"],
            conversation_id=conversation_id
        )
    
    patient = result["patient"]
    ecg_data = result["ecg_data"]
    
    # Format date and time
    date_str = f"{ecg_data['study_date'][:4]}-{ecg_data['study_date'][4:6]}-{ecg_data['study_date'][6:]}"
    time_str = f"{ecg_data['study_time'][:2]}:{ecg_data['study_time'][2:4]}"
    
    # Generate report content
    report_lines = [
        "Diagnostic Report",
        "=================",
        "",
        "Patient Information:",
        f"- Name: {patient['name']}",
        f"- ID: {patient['patient_id']}",
        f"- Gender: {patient.get('gender', 'Unknown')}",
        f"- Birth Date: {patient.get('birth_date', 'Unknown')}"
    ]
    
    # Add ECG data
    report_lines.extend([
        "",
        "ECG Data:",
        f"- Study Date: {date_str}",
        f"- Study Time: {time_str}"
    ])
    
    # Add waveform info if available
    waveform = ecg_data.get("waveform", {})
    if waveform:
        report_lines.extend([
            "",
            "Waveform Information:",
            f"- Sampling Rate: {waveform.get('sampling_rate', 'Unknown')} Hz",
            f"- Duration: {waveform.get('duration', 'Unknown')} seconds",
            f"- Leads: {', '.join(waveform.get('leads', []))}"
        ])
    
    # Add diagnosis if available
    diagnosis = ecg_data.get("diagnosis", {})
    if diagnosis:
        report_lines.extend([
            "",
            "Diagnostic Results:",
            f"- Heart Rate: {diagnosis.get('heart_rate', 'Not Available')} BPM",
            f"- Rhythm: {diagnosis.get('rhythm', 'Not Available')}",
            f"- Interpretation: {diagnosis.get('interpretation', 'Not Available')}"
        ])
        
        # Add probabilities if available
        probabilities = diagnosis.get('probabilities', {})
        if probabilities:
            report_lines.extend([
                "",
                "Probability of Conditions:"
            ])
            for condition, prob in probabilities.items():
                # Convert probability to float if it's a string
                try:
                    prob_value = float(prob) if isinstance(prob, str) else prob
                    report_lines.append(f"- {condition}: {prob_value:.1%}")
                except (ValueError, TypeError) as e:
                    logging.warning(f"[handle_generate_report] Invalid probability format for {condition}: {prob}")
                    report_lines.append(f"- {condition}: {prob}")
    
    return ChatResponse(
        message="\n".join(report_lines) + "\n\nIf you need more detailed analysis or have any questions, please let me know.",
        conversation_id=conversation_id,
        metadata={
            "ecg_data": ecg_data["waveform"],
            "patient_context": {
                "patient_id": patient_id,
                "patient_name": patient['name'],
                "study_date": date_str,
                "study_time": time_str
            }
        }
    )

async def handle_export_report(db, extracted_info: Dict[str, Any], conversation_id: str) -> ChatResponse:
    """Generate and export a PDF report for a patient's ECG data and diagnosis"""
    import os
    logging.info(f"[handle_export_report] Processing export report request with extracted_info: {extracted_info}")
    
    # Get instance_id from the session
    session = await session_manager.get_session(conversation_id)
    instance_id = None
    
    # First check if instance_id is provided in extracted_info
    if "instance_id" in extracted_info and extracted_info["instance_id"]:
        instance_id = extracted_info["instance_id"]
        logging.info(f"[handle_export_report] Using instance_id from extracted_info: {instance_id}")
    elif "ecg_id" in extracted_info and extracted_info["ecg_id"]:
        instance_id = extracted_info["ecg_id"]
        logging.info(f"[handle_export_report] Using ecg_id from extracted_info: {instance_id}")
    # If not in extracted_info, try to get from session
    elif session and hasattr(session, 'ecg_instance_id') and session.ecg_instance_id:
        instance_id = session.ecg_instance_id
        logging.info(f"[handle_export_report] Using instance_id from session: {instance_id}")
    # If still no instance_id, inform the user
    else:
        logging.warning("[handle_export_report] No instance_id found in extracted_info or session")
        return ChatResponse(
            message="No ECG instance ID found. Please view an ECG first or provide an ECG instance ID.",
            conversation_id=conversation_id
        )
    
    # Get patient ID, prioritize from extracted info
    patient_id = extracted_info.get("patient_id")
    if not patient_id:
        # Try to get from session
        session = await session_manager.get_session(conversation_id)
        if session and session.patient_id:
            patient_id = session.patient_id
            logging.info(f"[handle_export_report] Using patient_id from session: {patient_id}")
        else:
            logging.warning("[handle_export_report] No patient_id in extracted_info or session")
            return ChatResponse(
                message="Please specify a patient ID to generate a report.",
                conversation_id=conversation_id
            )
    
    # We already have the session and instance_id, no need to store it again if it came from the session
    # Only update if the instance_id came from extracted_info and not from the session
    if "instance_id" in extracted_info or "ecg_id" in extracted_info:
        try:
            if session:
                # Only update if this instance belongs to the current patient
                session.ecg_instance_id = instance_id
                await session_manager.save_session(session)
                logging.info(f"[handle_export_report] Updated instance_id {instance_id} in session for patient {patient_id}")
        except Exception as e:
            logging.error(f"[handle_export_report] Error updating session with instance_id: {str(e)}", exc_info=True)
    
    # Verify that the patient exists
    try:
        patient = await db.get_patient(patient_id)
        if not patient:
            return ChatResponse(
                message=f"Patient with ID {patient_id} not found.",
                conversation_id=conversation_id
            )
        logging.info(f"[handle_export_report] Found patient: {patient.name} (ID: {patient.patient_id})")
    except Exception as e:
        logging.error(f"[handle_export_report] Error getting patient: {str(e)}", exc_info=True)
        return ChatResponse(
            message=f"Error retrieving patient information: {str(e)}",
            conversation_id=conversation_id
        )
    
    # Get the diagnosis for the specific instance_id
    try:
        diagnosis = await get_latest_diagnosis(db, instance_id=instance_id)
        
        diseases = []
        
        # If we found a diagnosis
        if diagnosis:
            logging.info(f"[handle_export_report] Found diagnosis for patient {patient_id} and instance {instance_id}")
            
            # Extract diseases with probability > 0.03 from both models
            threshold = 0.03
            
            # Check original_dnn_model
            if "original_dnn_model" in diagnosis:
                for disease, prob in diagnosis["original_dnn_model"].items():
                    if float(prob) > threshold and disease not in diseases and disease is not 'NORM':
                        diseases.append(disease)
            
            # Check resnet_fine_tuned_model
            if "resnet_fine_tuned_model" in diagnosis:
                for disease, prob in diagnosis["resnet_fine_tuned_model"].items():
                    if float(prob) > threshold and disease not in diseases and disease is not 'NORM':
                        diseases.append(disease)
            
          
    
    except Exception as e:
        logging.error(f"[handle_export_report] Error extracting diseases from diagnosis result: {str(e)}", exc_info=True)
        diseases = []
    
    
    # If still no diseases, return message
    if not diseases:
        logging.warning(f"[handle_export_report] No diseases found for analysis for instance {instance_id}")
        return ChatResponse(
            message="No diseases detected for analysis. Cannot generate report.",
            conversation_id=conversation_id
        )
    
    logging.info(f"[handle_export_report] Using diseases for analysis: {diseases}")
    logging.info(f"[handle_export_report] Generating report for instance {instance_id} with diseases: {diseases}")
    
    try:
        # First, get the interpretation data from the AI server
        import requests
        from pathlib import Path
        
        # Prepare AI server request
        ai_server_url = f"{os.getenv('AISERVER_URI')}/interprete/{instance_id}"
        
        # Get temporary token for authentication

        temp_token = get_temp_token()
        logging.info("[handle_export_report] Successfully obtained temporary token for interpretation request")
        
        # Prepare the form data
        form_data_tuples = []
        for disease in diseases:
            form_data_tuples.append(("disease", disease))
        
        # Add the temporary token for authentication
        form_data_tuples.append(("temp_token", temp_token))
        
        # Make the request to the AI server
        try:
            response = requests.post(
                ai_server_url,
                data=form_data_tuples,
                timeout=60
            )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse the JSON response
            ai_response = response.json()
            logging.info(f"[handle_export_report] Received interpretation response from AI server for instance {instance_id}")
            
            # Try to fetch the ECG file data for the given instance ID
            from app.model.ecg_file import EcgFile
            
            # Use the diagnosis result we already have from get_latest_diagnosis
            if diagnosis and "interpretation" not in ai_response:
                # Use the interpret_diagnosis function to get interpretation
                from app.utils.ecg_utils import interpret_diagnosis
                diagnosis_interpretation = interpret_diagnosis(diagnosis)
                
                # Add interpretation to the AI response
                ai_response["interpretation"] = diagnosis_interpretation.get("interpretation", "No significant abnormalities detected")
            
            # Define reports directory
            ECG_REPORTS_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports", "ecg"))
            ECG_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Generate a PDF report from the interpretation data
            pdf_filename = f"{instance_id}.pdf"
            pdf_path = ECG_REPORTS_DIR / pdf_filename
            
            # Always generate a new PDF report
            logging.info(f"[handle_export_report] Generating PDF report for instance {instance_id}")
            from app.utils.report_generator import generate_ecg_report_pdf
            generate_ecg_report_pdf(ai_response, pdf_path, instance_id)
            
            # Get the PDF download URL using the utility function
            from app.utils.pdf_utils import get_report_download_url
            pdf_url = get_report_download_url(pdf_filename)
            
            # Get patient information
            patient = await db.get_patient(patient_id)
            if not patient:
                patient_name = "Unknown Patient"
            else:
                patient_name = patient.name
            
            # Create response message with download link
            message = f"### ECG Report Generated Successfully\n\n"
            message += f"**Patient:** {patient_name} (ID: {patient_id})\n"
            message += f"**Instance ID:** {instance_id}\n"
            message += f"**Analysis:** {', '.join(diseases)}\n\n"
            message += f"[Download PDF Report]({pdf_url})\n\n"
            message += "The report includes patient information, ECG data, and diagnostic results."
            
            return ChatResponse(
                message=message,
                conversation_id=conversation_id,
                patient_context={
                    "patient_id": patient_id,
                    "patient_name": patient_name,
                    "instance_id": instance_id,
                    "report_url": pdf_url
                }
            )
            
        except requests.Timeout:
            logging.error(f"[handle_export_report] Interpretation request timed out for instance {instance_id}")
            return ChatResponse(
                message="Interpretation request timed out, please try again later.",
                conversation_id=conversation_id
            )
        except requests.HTTPError as e:
            logging.error(f"[handle_export_report] AI server returned error for instance {instance_id}: {e.response.status_code} - {e.response.text}")
            return ChatResponse(
                message=f"AI server error: {e.response.text}",
                conversation_id=conversation_id
            )
        except requests.RequestException as e:
            logging.error(f"[handle_export_report] Network error during interpretation request for instance {instance_id}: {str(e)}")
            return ChatResponse(
                message="Unable to connect to AI server, please check network connection.",
                conversation_id=conversation_id
            )
        except ValueError as e:
            logging.error(f"[handle_export_report] Invalid JSON response from AI server for instance {instance_id}: {str(e)}")
            return ChatResponse(
                message="Invalid response from AI server.",
                conversation_id=conversation_id
            )
            
    except Exception as e:
        import traceback
        logging.error(f"[handle_export_report] Error generating PDF report: {str(e)}")
        logging.error(f"[handle_export_report] Traceback: {traceback.format_exc()}")
        return ChatResponse(
            message=f"Failed to generate PDF report: {str(e)}",
            conversation_id=conversation_id
        )

async def handle_compare_patients(db, extracted_info: Dict[str, Any], conversation_id: str) -> ChatResponse:
    """Compare ECG records for the same patient"""
    patient_id = extracted_info.get("patient_id")
    if not patient_id:
        return ChatResponse(
            message="Please specify a patient ID to compare ECG records.",
            conversation_id=conversation_id
        )
    
    # Get patient info
    patient = await db.get_patient(patient_id)
    if not patient:
        return ChatResponse(
            message=f"No patient found with ID: {patient_id}",
            conversation_id=conversation_id
        )
    
    # Get all ECG records for the patient
    ecg_records = await db.get_all_ecg(patient_id)
    if not ecg_records:
        return ChatResponse(
            message=f"No ECG records found for patient: {patient.name}",
            conversation_id=conversation_id
        )
    
    # Get diagnoses for each ECG
    comparisons = []
    for ecg in ecg_records:
        # Try to get diagnosis by instance_id if available, otherwise we can't get it without patient_id
        instance_id = ecg.get('dicom_uid')
        if instance_id:
            diagnosis = await get_latest_diagnosis(db, instance_id=instance_id)
        else:
            diagnosis = None
        if diagnosis:
            # Format diagnostic results
            # Find the highest probability results
            top_findings = []
            
            # Check DNN model
            dnn_probs = diagnosis.get('original_dnn_model', {})
            if dnn_probs:
                max_dnn = max(dnn_probs.items(), key=lambda x: float(x[1]), default=(None, 0))
                if max_dnn[0] and float(max_dnn[1]) > 0.01:
                    top_findings.append(f"{max_dnn[0]}: {float(max_dnn[1])*100:.1f}%")
            
            # Check ResNet model
            resnet_probs = diagnosis.get('resnet_fine_tuned_model', {})
            if resnet_probs:
                max_resnet = max(resnet_probs.items(), key=lambda x: float(x[1]), default=(None, 0))
                if max_resnet[0] and float(max_resnet[1]) > 0.01:
                    top_findings.append(f"{max_resnet[0]}: {float(max_resnet[1])*100:.1f}%")
            
            # If no significant findings, show as normal
            summary = ", ".join(top_findings) if top_findings else "Normal"
            
            comparisons.append({
                "study_date": ecg.get('study_date', 'Unknown'),
                "study_time": ecg.get('study_time', 'Unknown'),
                "diagnosis_summary": summary
            })
    
    if not comparisons:
        return ChatResponse(
            message=f"No diagnostic data found for patient: {patient.name}",
            conversation_id=conversation_id
        )
    
    # Format comparison table
    table = [
        f"ECG Comparison for Patient: {patient.name}",
        "-" * 70,
        f"{'Date':10} | {'Time':8} | {'Findings'}",
        "-" * 70
    ]
    
    # Sort by date and time
    comparisons.sort(key=lambda x: (x['study_date'], x['study_time']))
    
    for c in comparisons:
        date = c['study_date'][:4] + '-' + c['study_date'][4:6] + '-' + c['study_date'][6:] if len(c['study_date']) == 8 else c['study_date']
        time = c['study_time'][:2] + ':' + c['study_time'][2:4] if len(c['study_time']) >= 4 else c['study_time']
        table.append(
            f"{date:10} | {time:8} | {c['diagnosis_summary']}"
        )
    
    return ChatResponse(
        message="\n".join(table),
        conversation_id=conversation_id,
        patient_context={
            "patient_id": patient_id,
            "ecg_records": ecg_records
        }
    )

async def handle_medical_question(extracted_info: Dict[str, Any], conversation_id: str, original_input: str = None) -> ChatResponse:
    """Handle general medical questions using OpenAI"""
    # Get patient info (question is no longer in the prompt)
    patient_id = extracted_info.get("patient_id")
    # We no longer have question or response in the prompt
    question = original_input if original_input else ""  # Use original_input directly
    response = None
    
    # Always use original input if provided (this will contain the original URLs)
    if original_input:
        logging.info(f"[handle_medical_question] Original input: {original_input}")
        logging.info(f"[handle_medical_question] Extracted question: {question}")
        
        # Simply use the original input as the question
        question = original_input
        logging.info(f"[handle_medical_question] Using original input directly: {question}")
    
    if not question:
        return ChatResponse(
            message="Please describe your medical question.",
            conversation_id=conversation_id
        )
    
    # If preprocessed response exists, return it
    if response:
        return ChatResponse(
            message=response,
            conversation_id=conversation_id,
            patient_context={"patient_id": patient_id} if patient_id else None
        )
    
    # Get patient context if available
    patient_info = "No patient context available"
    if patient_id:
        try:
            patient = await db.get_patient(patient_id)
            if patient:
                patient_info = f"Patient: {patient.name}, ID: {patient.patient_id}, "
                patient_info += f"Gender: {patient.gender if patient.gender else 'Not specified'}"
                patient_info += f", Birth Date: {patient.birth_date.strftime('%Y-%m-%d') if patient.birth_date else 'Not specified'}"
        except Exception as e:
            logging.error(f"Error getting patient info: {e}")
    
    # Set image_info to empty string as we're using the original question directly
    image_info = ""
    
    # Create a new LLM chain to handle the question
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        chain = LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_messages([
                ("system", """You are a professional medical assistant with extensive knowledge in various medical fields.
                Please provide professional and accurate answers based on the user's questions.
                
                When answering, please follow these guidelines:
                1. Use professional yet understandable language
                2. Provide relevant medical reference data when necessary
                3. Suggest further examinations if needed
                4. Recommend immediate medical attention for urgent cases
                5. For ECG-related questions:
                   - Explain the significance of findings
                   - Discuss potential implications
                   - Mention typical reference ranges
                   - Suggest follow-up tests if needed
                6. For image analysis:
                   - IMPORTANT: If the user has shared image URLs, ASSUME these are medical images (likely ECG or other diagnostic images)
                   - If you see URLs in the user's question, acknowledge them and explain what type of medical image they likely are
                   - For ECG images, explain that they typically show heart rhythm patterns and electrical activity
                   - For images with 'aflt' or 'multilead' in the URL, explain these are likely atrial flutter or multi-lead ECG recordings
                   - Explain that while you cannot directly view the images, the user should use the system's ECG analysis features
                
                Current patient context: {patient_info}{image_info}
                """),
                ("human", "{question}")
            ])
        )
        
        # Simply invoke the chain with the original question
        response = await chain.ainvoke({"question": question, "patient_info": patient_info, "image_info": image_info})
        message = response["text"]
        
        return ChatResponse(
            message=message,
            conversation_id=conversation_id,
            patient_context={"patient_id": patient_id} if patient_id else None
        )
        
    except Exception as e:
        logging.error(f"Error handling medical question: {e}")
        return ChatResponse(
            message="Sorry, I encountered an error processing your question. Please try again or ask a different question.",
            conversation_id=conversation_id,
            patient_context={"patient_id": patient_id} if patient_id else None
        )
