import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid
import re
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage
from app.data.memory.session import SessionManager
from app.core.database import Database
from app.utils.models import ChatMessage, ChatResponse, Memory, SessionData
from app.model.patient import Patient
from app.handlers.intents import Intent, IntentResult, handle_switch_patient, \
                   handle_patient_query, handle_analyze_ecg, handle_view_ecg, \
                   handle_generate_report, handle_compare_patients, handle_medical_question
from app.utils.tools import (
    get_patient_data, get_ecg_plot,
    analyze_ecg, search_medical_knowledge, recommend_treatment
)
from app.core.intent_classifier import IntentClassifier

class DoctorAgent:
    def __init__(self):
        # Get API key from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key
        )
        self.db = Database()
        self.session_manager = SessionManager()
        self.conversations: Dict[str, List[ChatMessage]] = {}
        self.patient_context: Dict[str, str] = {}
        self.intent_classifier = IntentClassifier()
        
        # Define tools
        self.tools = [
            Tool(
                name="get_patient_info",
                func=get_patient_data,
                description="Get patient basic information and diagnosis summary"
            ),
            Tool(
                name="view_ecg",
                func=get_ecg_plot,
                description="Get and display patient's ECG waveform data"
            ),
            Tool(
                name="analyze_ecg",
                func=analyze_ecg,
                description="Analyze ECG data and get detailed diagnostic results"
            ),
            Tool(
                name="search_knowledge",
                func=search_medical_knowledge,
                description="Search relevant medical knowledge"
            ),
            Tool(
                name="recommend_treatment",
                func=recommend_treatment,
                description="Recommend treatment plan based on diagnosis results"
            )
        ]
        
        # Define prompt template
        self.prompt = PromptTemplate(
            input_variables=["history", "human_input", "memory", "tools", "patient_context"],
            template="""You are a professional AI medical assistant, specialized in helping doctors analyze ECG data and provide diagnostic recommendations.

Current Patient Information:
{patient_context}

Conversation History:
{history}

Relevant Memories:
{memory}

Available Tools:
{tools}

Please use appropriate tools to answer questions based on patient information and context.

Human: {human_input}
AI Doctor: Let me help you analyze this case."""
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=True
        )
    
    async def _get_patient_context(self, patient_id: str) -> str:
        """Get patient context information"""
        logging.info(f"[DoctorAgent._get_patient_context] Fetching patient info for ID: {patient_id}")
        if not patient_id:
            return "No patient specified"
            
        patient = await self.db.get_patient(patient_id)
        if patient:
            logging.info(f"[DoctorAgent._get_patient_context] Found patient: {patient.name}")
        else:
            logging.warning(f"[DoctorAgent._get_patient_context] Patient not found for ID: {patient_id}")
            return f"Patient not found (ID: {patient_id})"
            
        diagnosis = await self.db.get_diagnosis(patient_id)
        context = f"Patient ID: {patient.patient_id}\n"
        context += f"Name: {patient.name}\n"
        context += f"Gender: {patient.gender.value if patient.gender else 'Unknown'}\n"
        context += f"Birth Date: {patient.birth_date.strftime('%Y-%m-%d') if patient.birth_date else 'Unknown'}\n"
        
        if diagnosis:
            context += f"Primary Diagnosis: {diagnosis.primary_diagnosis}\n"
            context += f"Confidence: {diagnosis.confidence}\n"
        
        return context
    
    async def chat(self, message: str, conversation_id: Optional[str] = None, patient_id: Optional[str] = None) -> ChatResponse:
        """Process a chat message and return a response"""
        try:
            # Ensure we have a valid conversation_id
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            logging.info(f"[DoctorAgent.chat] Starting chat with message: {message}, conversation_id: {conversation_id}, patient_id: {patient_id}")
            
            # Initialize conversation history if needed
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
            
            # Add user message to history
            self.conversations[conversation_id].append(
                ChatMessage(role="user", content=message)
            )
            
            # Get current session data
            session = await self.session_manager.get_session(conversation_id)
            if not patient_id and session:
                patient_id = session.patient_id
            
            # Use IntentClassifier to identify intent
            intent_result = await self.intent_classifier.classify(
                text=message,
                current_patient_id=patient_id
            )
            
            if not intent_result:
                return ChatResponse(
                    message="Sorry, I couldn't understand your intent. Please rephrase your request.",
                    conversation_id=conversation_id
                )
            
            logging.info(f"[DoctorAgent.chat] Detected intent: {intent_result.intent} with confidence: {intent_result.confidence}")
            
            # Handle the intent
            try:
                if intent_result.intent == Intent.QUERY_PATIENT:
                    # Search and load patient information
                    response = await handle_patient_query(intent_result.extracted_info, conversation_id)
                    if response and response.patient_context and 'patient_id' in response.patient_context:
                        # If patient found, update session context
                        session = SessionData(session_id=conversation_id)
                        session.patient_id = response.patient_context['patient_id']
                        await self.session_manager.save_session(session)
                elif intent_result.intent == Intent.SWITCH_PATIENT:
                    # Switch current patient
                    response = await handle_switch_patient(intent_result.extracted_info, conversation_id)
                    if response and response.patient_context and 'patient_id' in response.patient_context:
                        session = SessionData(session_id=conversation_id)
                        session.patient_id = response.patient_context['patient_id']
                        await self.session_manager.save_session(session)
                elif intent_result.intent == Intent.VIEW_ECG:
                    # View ECG waveform data
                    response = await handle_view_ecg(self.db, intent_result.extracted_info, conversation_id)
                elif intent_result.intent == Intent.ANALYZE_ECG:
                    # Analyze ECG or view analysis results
                    response = await handle_analyze_ecg(self.db, intent_result.extracted_info, conversation_id)
                elif intent_result.intent == Intent.GENERATE_REPORT:
                    # Generate diagnostic report
                    response = await handle_generate_report(self.db, intent_result.extracted_info, conversation_id)
                elif intent_result.intent == Intent.MEDICAL_QUESTION:
                    # Answer medical questions
                    response = await handle_medical_question(intent_result.extracted_info, conversation_id)
                else:
                    response = ChatResponse(
                        message="Sorry, I cannot handle this type of request at the moment.",
                        conversation_id=conversation_id
                    )
                
                # Save assistant's response to conversation history
                self.conversations[conversation_id].append(
                    ChatMessage(role="assistant", content=response.message)
                )
                
                return response
                
            except Exception as e:
                logging.error(f"[DoctorAgent.chat] Error handling intent: {str(e)}", exc_info=True)
                return ChatResponse(
                    message="Sorry, an error occurred while processing your request. Please try again or rephrase your request.",
                    conversation_id=conversation_id
                )
                
        except Exception as e:
            logging.error(f"[DoctorAgent.chat] Error occurred: {str(e)}", exc_info=True)
            raise e

    def get_conversation_history(self, conversation_id: str) -> List[ChatMessage]:
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation not found (ID: {conversation_id})")
        return self.conversations[conversation_id]
