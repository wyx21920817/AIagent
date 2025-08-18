from typing import Dict, Any, Optional
import json
import logging
import aiohttp
import os
from dotenv import load_dotenv
from app.handlers.intents import Intent, IntentResult

# Configure logging
app_logger = logging.getLogger('app')

# Load environment variables
load_dotenv()

class ApiIntentClassifier:
    """
    Intent classifier that uses an external API for classification.
    Makes POST requests to a specified endpoint to generate responses.
    """
    
    def __init__(self):
        """Initialize the API intent classifier with the endpoint URL."""
        # Get API endpoint from environment variables or use default
        self.api_url = os.getenv("INTENT_API_URL", "http://184.145.155.117:26253/api/generate")
        self.model = os.getenv("INTENT_API_MODEL", "gpt-oss:20b")
        app_logger.info(f"[ApiIntentClassifier] Initialized with API URL: {self.api_url}")
        
    async def classify(self, text: str, current_patient_id: Optional[str] = None) -> Optional[IntentResult]:
        """
        Classify the user's input text into an intent using the external API.
        
        Args:
            text: The user's input text
            current_patient_id: Optional current patient ID from session
            
        Returns:
            IntentResult object or None if classification fails
        """
        app_logger.info(f"[ApiIntentClassifier.classify] Input text: {text}")
        
        try:
            # Optimized prompt for faster response
            prompt = f"""Classify this medical assistant input into JSON format. Respond ONLY with the JSON object.

JSON FORMAT:
{{
    "intent": "<INTENT_TYPE>",
    "confidence": <CONFIDENCE_SCORE>,
    "extracted_info": {{
        "patient_id": "<PATIENT_ID or empty string>",
        "patient_name": "<PATIENT_NAME or empty string>",
        "question": "<MEDICAL_QUESTION or empty string>",
        "ecg_id": "<ECG_ID or empty string>",
        "diagnosis": "<DIAGNOSIS or empty string>"
    }}
}}

INTENT TYPES:
"query_patient": Search/info about patient
"switch_patient": Switch to another patient
"analyze_ecg": ECG analysis/results
"view_ecg": View ECG data
"medical_question": General medical questions
"generate_report": Generate report
"export_report": Export/download PDF report
"compare_patients": Compare patient data

Input: {text}
Patient context: {current_patient_id if current_patient_id else "None"}
"""
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            app_logger.info(f"[ApiIntentClassifier.classify] Sending request to API")
            
            # Make the API request
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload) as response:
                    if response.status != 200:
                        app_logger.error(f"[ApiIntentClassifier.classify] API request failed with status {response.status}")
                        return None
                    
                    # Parse the API response
                    api_response = await response.json()
                    app_logger.info(f"[ApiIntentClassifier.classify] Received API response")
                    
                    # Extract the generated text from the API response
                    if "response" not in api_response:
                        app_logger.error(f"[ApiIntentClassifier.classify] Missing 'response' in API response")
                        return None
                    
                    generated_text = api_response["response"]
                    app_logger.debug(f"[ApiIntentClassifier.classify] Generated text: {generated_text}")
                    
            # Try to extract JSON from the generated text
            try:
                # Find JSON content in the response (it might be embedded in other text)
                json_start = generated_text.find('{')
                json_end = generated_text.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    app_logger.error(f"[ApiIntentClassifier.classify] No JSON found in response")
                    return None
                
                json_content = generated_text[json_start:json_end]
                result = json.loads(json_content)
                app_logger.info(f"[ApiIntentClassifier.classify] Parsed result: {result}")
                
                # Validate required fields
                required_fields = ["intent", "confidence", "extracted_info"]
                missing_fields = [field for field in required_fields if field not in result]
                if missing_fields:
                    app_logger.error(f"Missing required fields: {missing_fields}")
                    return None
                
                # The API should now return intents that match our application's Intent enum
                # But we'll keep a mapping for backward compatibility or in case the API returns different values
                intent_mapping = {
                    # Direct mappings from API to our application intents
                    "query_patient": Intent.QUERY_PATIENT,
                    "switch_patient": Intent.SWITCH_PATIENT,
                    "analyze_ecg": Intent.ANALYZE_ECG,
                    "view_ecg": Intent.VIEW_ECG,
                    "medical_question": Intent.MEDICAL_QUESTION,
                    "generate_report": Intent.GENERATE_REPORT,
                    "export_report": Intent.EXPORT_REPORT,
                    "compare_patients": Intent.COMPARE_PATIENTS,
                    "other": Intent.OTHER,
                    
                    # Legacy/alternative mappings (in case the API returns these)
                    "greeting": Intent.OTHER,
                    "patient_search": Intent.QUERY_PATIENT,
                    "patient_info": Intent.QUERY_PATIENT,
                    "ecg_analysis": Intent.ANALYZE_ECG,
                    "diagnosis_info": Intent.ANALYZE_ECG,
                    "help": Intent.OTHER,
                    "goodbye": Intent.OTHER,
                    "unknown": Intent.OTHER
                }
                
                api_intent = result["intent"].lower()
                if api_intent in intent_mapping:
                    intent = intent_mapping[api_intent]
                    app_logger.info(f"[ApiIntentClassifier.classify] Mapped API intent '{api_intent}' to application intent '{intent}'")
                else:
                    try:
                        # Try direct mapping if not in our mapping dictionary
                        intent = Intent(api_intent)
                    except ValueError:
                        app_logger.error(f"Invalid intent: {api_intent}. No mapping available.")
                        return None
                
                # Validate confidence
                try:
                    confidence = float(result["confidence"])
                    if not 0 <= confidence <= 1:
                        app_logger.error(f"Invalid confidence value: {confidence}")
                        return None
                except (ValueError, TypeError):
                    app_logger.error(f"Invalid confidence format: {result.get('confidence')}")
                    return None
                
                # Handle extracted information
                extracted_info = result["extracted_info"]
                # Ensure all required fields exist
                required_info_fields = ["patient_id", "patient_name", "question", "ecg_id", "diagnosis"]
                for field in required_info_fields:
                    if field not in extracted_info:
                        extracted_info[field] = ""
                
                # If current patient ID exists and extracted info doesn't have patient ID, use current patient ID
                if current_patient_id and not extracted_info["patient_id"]:
                    extracted_info["patient_id"] = current_patient_id
                
                # Create intent result
                intent_result = IntentResult(
                    intent=intent,
                    confidence=confidence,
                    extracted_info=extracted_info
                )
                
                app_logger.info(f"[ApiIntentClassifier.classify] Created intent result: {intent_result.__dict__}")
                return intent_result
                
            except json.JSONDecodeError as e:
                app_logger.error(f"JSON parse error: {str(e)}\nContent: {generated_text}")
                return None
            
        except Exception as e:
            app_logger.error(f"Classification error: {str(e)}", exc_info=True)
            return None
