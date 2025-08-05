import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from app.utils.models import SessionData

class SessionManager:
    _instance = None
    _sessions: Dict[str, SessionData] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logging.info("[SessionManager] Created new singleton instance")
        return cls._instance
    
    def __init__(self):
        # 因为是单例，__init__可能被多次调用，但我们不想重置_sessions
        pass
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        try:
            session = self._sessions.get(session_id)
            if not session:
                return None
            
            # 确保获取到的是SessionData对象
            if not isinstance(session, SessionData):
                logging.error(f"[SessionManager] Invalid session type in storage: {type(session)}")
                self._sessions.pop(session_id)  # 移除无效的session
                return None
            
            # 检查session是否过期
            if (datetime.now() - session.last_access) >= timedelta(hours=24):
                logging.info(f"[SessionManager] Found expired session {session_id}, removing it")
                self._sessions.pop(session_id)
                return None
            
            # 更新访问时间
            session.last_access = datetime.now()
            logging.info(f"[SessionManager] Found valid session {session_id} with patient_id: {session.patient_id}")
            return session
        except Exception as e:
            logging.error(f"[SessionManager] Error getting session: {e}")
            return None
    
    async def save_session(self, session: SessionData):
        try:
            # 确保传入的是SessionData对象
            if not isinstance(session, SessionData):
                logging.error(f"[SessionManager] Invalid session type: {type(session)}")
                return
            
            # 检查现有session
            existing_session = self._sessions.get(session.session_id)
            if existing_session and isinstance(existing_session, SessionData):
                # 如果现有session有patient_id，且新session没有，保留现有的
                if existing_session.patient_id and not session.patient_id:
                    session.patient_id = existing_session.patient_id
                    logging.info(f"[SessionManager] Kept existing patient_id: {session.patient_id} for session {session.session_id}")
            
            # 更新访问时间
            session.last_access = datetime.now()
            
            # 保存session
            self._sessions[session.session_id] = session
            logging.info(f"[SessionManager] Saved session {session.session_id} with patient_id: {session.patient_id}")
        except Exception as e:
            logging.error(f"[SessionManager] Error saving session: {e}")
    
    async def delete_session(self, session_id: str):
        if session_id in self._sessions:
            self._sessions.pop(session_id)
            logging.info(f"[SessionManager] Deleted session {session_id}")
    
    async def cleanup_expired_sessions(self):
        """清理过期会话"""
        now = datetime.now()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.last_access > timedelta(hours=24)
        ]
        for sid in expired:
            self.sessions.pop(sid)
            logging.info(f"[SessionManager] Cleaned up expired session {sid}")
