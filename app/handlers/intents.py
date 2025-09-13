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

async def get_latest_diagnosis(db: Database, patient_id: str, study_date: Optional[str] = None) -> Optional[Dict]:
    """Get the latest diagnosis for a patient, optionally filtered by study date"""
    try:
        diagnoses = await db.get_patient_diagnoses(patient_id)
        if not diagnoses:
            logging.info(f"[get_latest_diagnosis] No diagnoses found for patient {patient_id}")
            return None
            
        # Filter out diagnoses without timestamps
        valid_diagnoses = []
        for diag in diagnoses:
            if not isinstance(diag, dict):
                logging.warning(f"[get_latest_diagnosis] Invalid diagnosis format for patient {patient_id}: {diag}")
                continue
                
            # If no diagnosis time, use record creation time
            if "diagnosis_time" not in diag and "created_at" in diag:
                diag["diagnosis_time"] = diag["created_at"]
                
            if "diagnosis_time" in diag:
                # If date specified, only keep diagnoses from that date
                if study_date:
                    diag_date = diag["diagnosis_time"][:8]  # Take first 8 characters as date
                    if diag_date == study_date:
                        valid_diagnoses.append(diag)
                else:
                    valid_diagnoses.append(diag)
            else:
                logging.warning(f"[get_latest_diagnosis] Diagnosis missing timestamp for patient {patient_id}: {diag}")
        
        if not valid_diagnoses:
            logging.warning(f"[get_latest_diagnosis] No valid diagnoses found for patient {patient_id}")
            return None
            
        # Sort by diagnosis time to get the latest
        logging.info(f"[get_latest_diagnosis] Found {len(valid_diagnoses)} valid diagnoses for patient {patient_id}")
        return sorted(valid_diagnoses, key=lambda x: x["diagnosis_time"])[-1]
        
    except Exception as e:
        logging.error(f"[get_latest_diagnosis] Failed to get diagnosis for patient {patient_id}: {str(e)}", exc_info=True)
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
    
    # Get latest diagnosis if available
    try:
        diagnosis = await get_latest_diagnosis(db, patient.patient_id)
        if diagnosis:
            info.extend([
                "",
                "Latest Diagnosis:",
                f"Date: {diagnosis.get('diagnosis_time', 'Unknown')}",
                f"Heart Rate: {diagnosis.get('heart_rate', 'N/A')} bpm",
                f"Rhythm: {diagnosis.get('rhythm', 'N/A')}",
                f"Interpretation: {diagnosis.get('interpretation', 'N/A')}"
            ])
    except Exception as e:
        logging.error(f"[handle_patient_query] Error getting diagnosis: {str(e)}")
    
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
    
    # Get study date if specified
    study_date = extracted_info.get("study_date")
    
    # Get ECG data using the utility function
    from app.utils.ecg_utils import get_patient_ecg_data
    result = get_patient_ecg_data(patient_id, study_date)
    
    if "error" in result:
        logging.warning(f"[handle_analyze_ecg] {result['error']} for patient {patient_id}")
        return ChatResponse(
            message=result["error"],
            conversation_id=conversation_id
        )
    
    # Get ECG data
    ecg_data = result["ecg_data"]
    
    # Get the diagnosis_result directly from the raw_results field
    # This is the format you showed in your example with original_dnn_model and resnet_fine_tuned_model
    diagnosis_result = ecg_data.get("diagnosis", {}).get("raw_results", {})
    
    # Check if we have any diagnosis data
    if not diagnosis_result:
        logging.warning(f"[handle_analyze_ecg] No diagnosis data found for patient {patient_id}")
        return ChatResponse(
            message="No diagnosis data found for this patient.",
            conversation_id=conversation_id
        )
    
    # Log the diagnosis_result for debugging
    logging.info(f"[handle_analyze_ecg] Diagnosis result: {diagnosis_result}")
    
    # Format diagnosis probabilities directly from the diagnosis_result
    prob_lines = []
    
    # DNN model probabilities (Arrhythmia Analysis)
    dnn_probs = probabilities.get('original dnn model', {})
    if dnn_probs:
        arrhythmia_types = {
            '1dAVb': 'First-degree AV block',
            'RBBB': 'Right Bundle Branch Block',
            'LBBB': 'Left Bundle Branch Block',
            'SB': 'Sinus Bradycardia',
            'AF': 'Atrial Fibrillation',
            'ST': 'Sinus Tachycardia'
        }
        prob_lines.append("\nArrhythmia Analysis:")
        for condition, prob in dnn_probs.items():
            if isinstance(prob, (int, float)) and float(prob) > 0.01:  # Only show if probability > 1%
                condition_name = arrhythmia_types.get(condition, condition)
                prob_lines.append(f"- {condition_name}: {float(prob):.1%}")
    
    # ResNet model probabilities (Morphological Analysis)
    resnet_probs = probabilities.get('resnet fine tuned model', {})
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
    
    # Generate a simple interpretation based on the diagnosis_result
    formatted_interpretation = "- No significant abnormalities detected"
    
    # Check if we have significant findings in DNN model
    dnn_probs = diagnosis_result.get('original_dnn_model', {})
    if dnn_probs:
        max_dnn_key = max(dnn_probs.items(), key=lambda x: float(x[1]))[0]
        max_dnn_prob = float(dnn_probs[max_dnn_key])
        if max_dnn_prob > 0.01:
            formatted_interpretation = f"- Rhythm Analysis: {max_dnn_key} (Confidence: {max_dnn_prob*100:.1f}%)"
    
    # Check if we have significant findings in ResNet model
    resnet_probs = diagnosis_result.get('resnet_fine_tuned_model', {})
    if resnet_probs:
        max_resnet_key = max(resnet_probs.items(), key=lambda x: float(x[1]))[0]
        max_resnet_prob = float(resnet_probs[max_resnet_key])
        if max_resnet_prob > 0.01:
            if formatted_interpretation == "- No significant abnormalities detected":
                formatted_interpretation = f"- Morphology Analysis: {max_resnet_key} (Confidence: {max_resnet_prob*100:.1f}%)"
            else:
                formatted_interpretation += f"\n- Morphology Analysis: {max_resnet_key} (Confidence: {max_resnet_prob*100:.1f}%)"
    
    return ChatResponse(
        message=f"ECG Analysis for {result['patient']['name']}:\n\n" + 
                "Diagnostic Results:\n" + 
                f"Interpretation:\n{formatted_interpretation}\n\n" + 
                "Model Analysis Results:\n" + 
                "\n".join(prob_lines),
        conversation_id=conversation_id,
        metadata={
            "diagnosis": diagnosis,
            "probabilities": probabilities,
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
    
    # Get study date if specified
    study_date = extracted_info.get("study_date")
    
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
    
    # Add diagnosis if available
    diagnosis_info = ecg_data.get('diagnosis', {})
    diagnosis_result = diagnosis_info.get('raw_results', {})
    
    if diagnosis_result:
        # Generate interpretation based on diagnosis_result
        formatted_interpretation = "- No significant abnormalities detected"
        
        # Check if we have significant findings in DNN model
        dnn_probs = diagnosis_result.get('original_dnn_model', {})
        if dnn_probs:
            max_dnn_key = max(dnn_probs.items(), key=lambda x: float(x[1]))[0]
            max_dnn_prob = float(dnn_probs[max_dnn_key])
            if max_dnn_prob > 0.01:
                formatted_interpretation = f"- Rhythm Analysis: {max_dnn_key} (Confidence: {max_dnn_prob*100:.1f}%)"
        
        # Check if we have significant findings in ResNet model
        resnet_probs = diagnosis_result.get('resnet_fine_tuned_model', {})
        if resnet_probs:
            max_resnet_key = max(resnet_probs.items(), key=lambda x: float(x[1]))[0]
            max_resnet_prob = float(resnet_probs[max_resnet_key])
            if max_resnet_prob > 0.01:
                if formatted_interpretation == "- No significant abnormalities detected":
                    formatted_interpretation = f"- Morphology Analysis: {max_resnet_key} (Confidence: {max_resnet_prob*100:.1f}%)"
                else:
                    formatted_interpretation += f"\n- Morphology Analysis: {max_resnet_key} (Confidence: {max_resnet_prob*100:.1f}%)"
        
        # Add diagnostic results header
        info_lines.extend([
            "",
            "Diagnostic Results:",
            "Interpretation:",
            formatted_interpretation
        ])
        
        # Add model analysis results header
        info_lines.extend([
            "",
            "Model Analysis Results:"
        ])
        
        # DNN model (Arrhythmia Analysis)
        dnn_probs = diagnosis.get('original_dnn_model', diagnosis.get('original dnn model', {}))
        if dnn_probs:
            # Convert all values to float for comparison
            float_probs = {k: float(v) for k, v in dnn_probs.items()}
            
            # Find the maximum probability
            if float_probs:
                max_key = max(float_probs, key=float_probs.get)
                max_prob = float_probs[max_key]
                
                # Only add arrhythmia analysis if there's a significant probability
                if max_prob > 0.01:
                    info_lines.append("Arrhythmia Analysis:")
                    info_lines.append(f"- {max_key}: {max_prob*100:.1f}%")
                else:
                    # If all probabilities are very low, indicate normal rhythm
                    info_lines.append("Arrhythmia Analysis:")
                    info_lines.append("- Normal Sinus Rhythm (inferred from low arrhythmia probabilities)")
            else:
                info_lines.append("Arrhythmia Analysis: No data available")
        
        # ResNet model (Morphological Analysis)
        resnet_probs = diagnosis_result.get('resnet_fine_tuned_model', diagnosis_result.get('resnet fine tuned model', {}))
        if resnet_probs:
            # Convert all values to float for comparison
            float_probs = {k: float(v) for k, v in resnet_probs.items()}
            
            # Find the maximum probability
            if float_probs:
                max_key = max(float_probs, key=float_probs.get)
                max_prob = float_probs[max_key]
                
                # Format probability based on its range
                if max_prob > 0.01:  # Only show if probability > 1%
                    info_lines.append("Morphological Analysis:")
                    # Format the probability display
                    prob_display = f"{max_prob*100:.1f}%"
                    info_lines.append(f"- {max_key}: {prob_display}")
            else:
                info_lines.append("Morphological Analysis: No data available")
    
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
    
    # Get study date if specified
    study_date = extracted_info.get("study_date")
    
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
    
    # Extract instance ID from the question if available
    question = extracted_info.get("question", "")
    instance_id = extracted_info.get("instance_id") or extracted_info.get("ecg_id")
    
    # Try to extract instance ID from the question if not found in extracted_info
    if not instance_id and question:
        import re
        # Look for patterns like "ECG instance ID is X" or "instance ID X"
        id_patterns = [
            r"ECG instance ID is ([\w\-]+)",
            r"instance ID is ([\w\-]+)",
            r"instance ID ([\w\-]+)",
            r"ECG ID is ([\w\-]+)",
            r"ECG ID ([\w\-]+)"
        ]
        
        for pattern in id_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                instance_id = match.group(1)
                logging.info(f"[handle_export_report] Extracted instance_id from question: {instance_id}")
                break
    
    # If still no instance_id, inform the user
    if not instance_id:
        return ChatResponse(
            message="Please provide an ECG instance ID to generate a report.",
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
    
    # Try to find the ECG file by instance_id to extract diagnosis results
    try:
        # First try to find in the ecg_files collection by instanceIDs field
        from pymongo import MongoClient
        mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
        client = MongoClient(mongodb_uri)
        db_mongo = client.aiclinic
        
        # Look for ECG files that contain this instance_id in their instanceIDs array
        ecg_file = db_mongo.ecg_files.find_one({"dicom_uid": instance_id})
        
        # If not found, try looking by dicom_uid
        if not ecg_file:
            ecg_file = db_mongo.ecg_files.find_one({"dicom_uid": instance_id})
        
        diseases = []
        
        # If we found an ECG file with diagnosis results
        if ecg_file and "diagnosis_result" in ecg_file:
            diagnosis_result = ecg_file["diagnosis_result"]
            logging.info(f"[handle_export_report] Found diagnosis result for instance {instance_id}")
            
            # Extract diseases with probability > 0.03 from both models
            threshold = 0.03
            
            # Check original_dnn_model
            if "original_dnn_model" in diagnosis_result:
                for disease, prob in diagnosis_result["original_dnn_model"].items():
                    if prob > threshold and disease not in diseases:
                        diseases.append(disease)
            
            # Check resnet_fine_tuned_model
            if "resnet_fine_tuned_model" in diagnosis_result:
                for disease, prob in diagnosis_result["resnet_fine_tuned_model"].items():
                    if prob > threshold and disease not in diseases:
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
            
            # Extract heart rate and rhythm information from diagnosis results if available
            if ecg_file and "diagnosis_result" in ecg_file:
                # Use the interpret_diagnosis function to get heart rate and rhythm
                from app.utils.ecg_utils import interpret_diagnosis
                diagnosis_interpretation = interpret_diagnosis(ecg_file["diagnosis_result"])
                
                # Add heart rate and rhythm to the AI response
                ai_response["heart_rate"] = diagnosis_interpretation.get("heart_rate", "Not Available")
                ai_response["rhythm"] = diagnosis_interpretation.get("rhythm", "Not Available")
            
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
        diagnosis = await get_latest_diagnosis(db, patient_id, ecg.get('study_date'))
        if diagnosis:
            comparisons.append({
                "study_date": ecg.get('study_date', 'Unknown'),
                "study_time": ecg.get('study_time', 'Unknown'),
                "heart_rate": diagnosis.get('heart_rate', 'N/A'),
                "rhythm": diagnosis.get('rhythm', 'N/A'),
                "interpretation": diagnosis.get('interpretation', 'N/A')
            })
    
    if not comparisons:
        return ChatResponse(
            message=f"No diagnostic data found for patient: {patient.name}",
            conversation_id=conversation_id
        )
    
    # Format comparison table
    table = [
        f"ECG Comparison for Patient: {patient.name}",
        "-" * 100,
        f"{'Date':10} | {'Time':8} | {'Heart Rate':10} | {'Rhythm':30} | {'Interpretation':30}",
        "-" * 100
    ]
    
    # Sort by date and time
    comparisons.sort(key=lambda x: (x['study_date'], x['study_time']))
    
    for c in comparisons:
        date = c['study_date'][:4] + '-' + c['study_date'][4:6] + '-' + c['study_date'][6:] if len(c['study_date']) == 8 else c['study_date']
        time = c['study_time'][:2] + ':' + c['study_time'][2:4] if len(c['study_time']) >= 4 else c['study_time']
        table.append(
            f"{date:10} | {time:8} | {c['heart_rate']:10} | {c['rhythm'][:30]:30} | {c['interpretation'][:30]:30}"
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
    # Get question and patient info
    question = extracted_info.get("question")
    patient_id = extracted_info.get("patient_id")
    response = extracted_info.get("response")
    
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
