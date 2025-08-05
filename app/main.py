import uuid
import logging
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# 配置日志
# 设置日志格式
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_format)

# 配置 root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)

# 创建主应用日志器
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime
from app.core.agent import DoctorAgent
from app.utils.models import ChatMessage, ChatResponse, SessionData
from app.model.patient import Patient
from app.utils.tools import ecg_tools
from app.data.memory.session import SessionManager
from app.core.database import Database
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Doctor Assistant",
    description="An intelligent medical assistant with ECG analysis and memory capabilities",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize components
doctor_agent = DoctorAgent()
session_manager = SessionManager()

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    patient_id: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "message": "分析这个病人的ECG结果",
                "patient_id": "P12345",
                "conversation_id": None
            }
        }

async def get_session(session_id: str):
    session = await session_manager.get_session(session_id)
    if not session:
        session = SessionData(session_id=session_id)
        await session_manager.save_session(session)
    return session

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )

@app.post("/api/chat", 
         response_model=ChatResponse,
         summary="Send a message to the AI doctor",
         description="Send a message to the AI doctor and get a response based on patient context.")
async def chat(request: ChatRequest):
    try:
        # 打印请求内容
        logger.info(f"[chat] Received request:\n" +
                   f"Message: {request.message}\n" +
                   f"Conversation ID: {request.conversation_id}\n" +
                   f"Patient ID: {request.patient_id}")

        # Generate new conversation_id if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        logger.info(f"Using conversation_id: {conversation_id}")
        
        # Handle initialization request
        if request.message == 'init':
            logger.info(f"Handling initialization request with conversation_id: {conversation_id}")
            return ChatResponse(
                message="",  # 空消息，不会显示
                conversation_id=conversation_id
            )
        
        # 只检查session是否存在，不创建新的
        session = await session_manager.get_session(conversation_id)
        if session:
            logger.info(f"[chat] Found existing session: {conversation_id} with patient_id: {session.patient_id}")
            patient_id = session.patient_id
        else:
            patient_id = None
            logger.info(f"[chat] No existing session found for conversation: {conversation_id}")
        
        # 如果request中提供了patient_id，使用它
        if request.patient_id:
            patient_id = request.patient_id
            logger.info(f"[chat] Using patient_id from request: {patient_id}")
        
        # 记录patient_id的来源
        if patient_id:
            logger.info(f"[chat] Using patient_id: {patient_id} from {'request' if request.patient_id else 'session'}")
        else:
            logger.info(f"[chat] No patient_id available in request or session")
        
        logger.info(f"[chat] Received request: message='{request.message}', conversation_id='{conversation_id}', patient_id='{patient_id}')")
        response = await doctor_agent.chat(
            message=request.message,
            conversation_id=conversation_id,
            patient_id=patient_id
        )

        logger.info(f"[chat] Got response from doctor_agent, type: {type(response)}")

        try:
            # 打印响应内容
            if isinstance(response, ChatResponse):
                # 创建一个不包含波形数据的上下文副本用于日志记录
                log_context = response.patient_context.copy() if response.patient_context else None
                if log_context and 'ecg_data' in log_context:
                    log_context['ecg_data'] = '<waveform data omitted>'
                
                # 过滤掉消息中的base64图片数据
                message_for_log = response.message
                if 'data:image' in message_for_log:
                    message_for_log = message_for_log.split('data:image')[0] + '<image data omitted>'
                
                logger.info(f"[chat] Sending ChatResponse:\n" +
                          f"Message: {message_for_log}\n" +
                          f"Conversation ID: {response.conversation_id}\n" +
                          f"Patient Context: {log_context}")
            else:
                logger.warning(f"[chat] Response is not ChatResponse but {type(response)}")
                if isinstance(response, dict):
                    logger.info(f"[chat] Response dict: {response}")
                else:
                    logger.info(f"[chat] Response str: {str(response)}")
        except Exception as log_error:
            logger.error(f"[chat] Error logging response: {str(log_error)}\n" +
                        f"Response type: {type(response)}\n" +
                        f"Response repr: {repr(response)}")

        # 返回 JSON 响应
        return response
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        # 返回错误响应
        error_response = ChatResponse(
            message="An unexpected error occurred. Please try again later.",
            conversation_id=conversation_id,
            patient_context=None
        )
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )

@app.get("/patient/{patient_id}",
        summary="Get patient data",
        description="Retrieve patient information and latest diagnosis.")
async def get_patient(patient_id: str):
    try:
        data = await ecg_tools.get_patient_data(patient_id)
        if "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])
        return data
    except Exception as e:
        logger.error(f"Error fetching patient data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ecg/{patient_id}",
        summary="Get ECG plot data",
        description="Retrieve ECG data for plotting.")
async def get_ecg(patient_id: str, duration: float = 10.0):
    try:
        data = await ecg_tools.get_ecg_plot_data(patient_id, duration)
        if "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])
        return data
    except Exception as e:
        logger.error(f"Error fetching ECG data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/diagnosis/{patient_id}",
        summary="Get diagnosis details",
        description="Get detailed ECG analysis and diagnosis results.")
async def get_diagnosis(patient_id: str):
    try:
        data = await ecg_tools.analyze_ecg(patient_id)
        if "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])
        return data
    except Exception as e:
        logger.error(f"Error fetching diagnosis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}",
        response_model=List[ChatMessage],
        summary="Get conversation history",
        description="Retrieve the message history for a specific conversation.")
async def get_conversation_history(conversation_id: str):
    try:
        logger.info(f"Fetching history for conversation: {conversation_id}")
        history = doctor_agent.get_conversation_history(conversation_id)
        return history
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching conversation history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health",
        summary="Health check",
        description="Check if the service is running properly.")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    chat_html = os.path.join(os.path.dirname(__file__), "static", "chat.html")
    return FileResponse(chat_html)


