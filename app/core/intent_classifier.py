from typing import Dict, Any, Optional, Tuple
import json
import logging
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from app.handlers.intents import Intent, IntentResult
from app.utils.models import ChatResponse

# Configure logging
# Completely disable MongoDB-related logs
for logger_name in ['pymongo', 'pymongo.topology', 'pymongo.connection', 'pymongo.monitoring']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
    logging.getLogger(logger_name).propagate = False

# Disable debug logs from external libraries
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

# Only show our application's logs
app_logger = logging.getLogger('app')
app_logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv()

class ApiIntentClassifier:
    def __init__(self):
        # Get API key from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        # Create LLM instance
        self.llm = ChatOpenAI(
            temperature=0,  # Use deterministic output
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key
        )
        
        # Create medical question chain
        self.medical_chain = LLMChain(
            llm=ChatOpenAI(
                temperature=0.7,
                model_name="gpt-3.5-turbo",
                openai_api_key=openai_api_key
            ),
            prompt=ChatPromptTemplate.from_messages([
                ("system", """You are a professional medical assistant specializing in ECG and cardiology.
                When answering questions:
                1. For ECG-related questions, provide professional explanations
                2. For diagnostic results, explain their medical significance and potential treatments
                3. Include specific values and reference ranges when available
                4. Suggest further examinations if needed
                
                Current patient context: {patient_info}
                """),
                ("human", "{question}")
            ])
        )
        
        # Create ECG analysis chain
        self.ecg_chain = LLMChain(
            llm=ChatOpenAI(
                temperature=0.7,
                model_name="gpt-3.5-turbo",
                openai_api_key=openai_api_key
            ),
            prompt=ChatPromptTemplate.from_messages([
                ("system", """You are a professional ECG analyst.
                Please analyze and interpret the patient's ECG data and diagnostic results.
                
                Current patient context: {patient_info}
                ECG data: {ecg_data}
                Diagnostic results: {diagnosis_result}
                
                When analyzing:
                1. Explain the significance of any abnormal findings
                2. Compare with normal reference ranges
                3. Discuss potential clinical implications
                4. Suggest follow-up tests or monitoring if needed
                5. Highlight any critical or urgent findings
                """),
                ("human", "{question}")
            ])
        )
        
        # Define the system prompt for intent classification
        # Define the system prompt template
        system_template = '''You are an AI medical assistant that helps classify user intents. You must respond with a valid JSON object.

STRICT RESPONSE FORMAT:
{{
    "intent": "<INTENT_TYPE>",
    "confidence": <CONFIDENCE_SCORE>,
    "extracted_info": {{
        "patient_id": "<PATIENT_ID>",
        "ecg_id": "<ECG_ID>"
    }}
}}

INTENT TYPES (use exactly one of these values):
- "query_patient": When user wants to search for a patient OR asks about patient information
- "switch_patient": When user explicitly wants to switch from current patient to another
- "analyze_ecg": When user requests ECG analysis or asks about ECG results
- "view_ecg": When user wants to view ECG data
- "medical_question": When user asks general medical questions
- "generate_report": When user requests a report generation
- "export_report": When user wants to export or download a report as PDF

RULES:
1. CONFIDENCE must be a number between 0.0 and 1.0
2. Empty or unknown fields in extracted_info must be empty strings ("")
3. DO NOT include any text outside the JSON object
4. DO NOT include any explanations or additional content
5. Ensure the JSON is properly formatted with correct quotes and braces

Example valid responses:
1. User: "Show me patient John's information"
{{
    "intent": "query_patient",
    "confidence": 0.95,
    "extracted_info": {{
        "patient_id": "123456",
        "ecg_id": ""
    }}
}}

2. User: "What's the ECG analysis for instance ID abc-123?"
{{
    "intent": "analyze_ecg",
    "confidence": 0.9,
    "extracted_info": {{
        "patient_id": "",
        "ecg_id": "abc-123"
    }}
}}
'''

        # Create messages template
        messages = [
            ("system", system_template),
            ("human", "{text}")
        ]
        logging.debug(f"[IntentClassifier.__init__] Creating prompt with messages: {messages}")
        self.prompt = ChatPromptTemplate.from_messages(messages)

    async def classify(self, text: str, current_patient_id: Optional[str] = None) -> Optional[IntentResult]:
        """Classify the user's input text into an intent
        
        Args:
            text: The user's input text
            current_patient_id: Optional current patient ID from session
        """
        try:
            # If current patient ID exists, add it to the text
            if current_patient_id:
                text = f"[Current Patient ID: {current_patient_id}] {text}"
            
            logging.info(f"[IntentClassifier.classify] Input text: {text}")
            
            try:
                # Get LLM's response
                logging.info(f"[IntentClassifier.classify] Prompt template: {self.prompt}")
                logging.info(f"[IntentClassifier.classify] Template variables: {{'text': {text}}}")
                
                messages = self.prompt.format_messages(text=text)
                logging.info(f"[IntentClassifier.classify] Formatted messages: {[{'role': m.type, 'content': m.content} for m in messages]}")
                
                response = await self.llm.ainvoke(messages)
                logging.info(f"[IntentClassifier.classify] Raw response: {response.content}")
            except Exception as e:
                logging.error(f"[IntentClassifier.classify] Error during message formatting: {str(e)}", exc_info=True)
                raise
            
            # Try to parse JSON response
            try:
                # Clean response content
                content = response.content.strip()
                # Remove any leading or trailing non-JSON content
                content = content[content.find('{'):content.rfind('}')+1]
                logging.debug(f"[IntentClassifier.classify] Cleaned content: {content}")
                
                result = json.loads(content)
                logging.info(f"[IntentClassifier.classify] Parsed result: {result}")
                
                # Validate required fields
                required_fields = ["intent", "confidence", "extracted_info"]
                missing_fields = [field for field in required_fields if field not in result]
                if missing_fields:
                    logging.error(f"Missing required fields: {missing_fields}")
                    return None
                
                # Validate intent
                try:
                    intent = Intent(result["intent"].lower())
                except ValueError:
                    logging.error(f"Invalid intent: {result.get('intent')}")
                    return None
                
                # Validate confidence
                try:
                    confidence = float(result["confidence"])
                    if not 0 <= confidence <= 1:
                        logging.error(f"Invalid confidence value: {confidence}")
                        return None
                except (ValueError, TypeError):
                    logging.error(f"Invalid confidence format: {result.get('confidence')}")
                    return None
                
                # Handle extracted information
                extracted_info = result["extracted_info"]
                # Ensure all required fields exist
                required_info_fields = ["patient_id", "ecg_id"]
                for field in required_info_fields:
                    if field not in extracted_info:
                        extracted_info[field] = ""
                        
                # Add empty values for backward compatibility
                extracted_info["patient_name"] = ""
                extracted_info["question"] = ""
                extracted_info["diagnosis"] = ""
                
                # If current patient ID exists and extracted info doesn't have patient ID, use current patient ID
                if current_patient_id and not extracted_info["patient_id"]:
                    extracted_info["patient_id"] = current_patient_id
                
                # Create intent result
                intent_result = IntentResult(
                    intent=intent,
                    confidence=confidence,
                    extracted_info=extracted_info
                )
                
                logging.info(f"[IntentClassifier.classify] Created intent result: {intent_result.__dict__}")
                return intent_result
                
            except json.JSONDecodeError as e:
                logging.error(f"JSON parse error: {str(e)}\nContent: {content}")
                return None
            
        except Exception as e:
            logging.error(f"Classification error: {str(e)}", exc_info=True)
            return None
            
   