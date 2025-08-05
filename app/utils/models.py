from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from datetime import datetime
from enum import Enum

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    patient_id: Optional[str] = None
    metadata: Optional[Dict] = None

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    patient_id: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    patient_context: Optional[Dict] = None  # 包含当前病人信息
    metadata: Optional[Dict] = None

class Memory(BaseModel):
    id: str
    patient_id: Optional[str] = None
    content: str
    type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict] = None

class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict
    required: List[str]

class SessionData(BaseModel):
    session_id: str
    patient_id: Optional[str] = None
    last_access: datetime = Field(default_factory=datetime.now)
    context: Dict = Field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SessionData':
        """从字典创建SessionData对象，确保类型正确"""
        if isinstance(data, dict):
            # 确保last_access是datetime类型
            if 'last_access' in data and isinstance(data['last_access'], str):
                data['last_access'] = datetime.fromisoformat(data['last_access'])
            return cls(**data)
        return data if isinstance(data, cls) else cls()
    
    def to_dict(self) -> Dict:
        """转换为字典，确保所有字段都可以正确序列化"""
        return {
            'session_id': self.session_id,
            'patient_id': self.patient_id,
            'last_access': self.last_access.isoformat(),
            'context': self.context
        }
