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
        # Since this is a singleton, __init__ may be called multiple times, but we don't want to reset _sessions
        pass
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        try:
            session = self._sessions.get(session_id)
            if not session:
                return None
            
            # Ensure we get a SessionData object
            if not isinstance(session, SessionData):
                logging.error(f"[SessionManager] Invalid session type in storage: {type(session)}")
                self._sessions.pop(session_id)  # Remove invalid session
                return None
            
            # Check if session has expired
            if (datetime.now() - session.last_access) >= timedelta(hours=24):
                logging.info(f"[SessionManager] Found expired session {session_id}, removing it")
                self._sessions.pop(session_id)
                return None
            
            # Update access time
            session.last_access = datetime.now()
            logging.info(f"[SessionManager] Found valid session {session_id} with patient_id: {session.patient_id}")
            return session
        except Exception as e:
            logging.error(f"[SessionManager] Error getting session: {e}")
            return None
    
    async def save_session(self, session: SessionData):
        try:
            # Ensure input is a SessionData object
            if not isinstance(session, SessionData):
                logging.error(f"[SessionManager] Invalid session type: {type(session)}")
                return
            
            # Check existing session
            existing_session = self._sessions.get(session.session_id)
            if existing_session and isinstance(existing_session, SessionData):
                # If patient_id changed, clear the ecg_instance_id
                if existing_session.patient_id and session.patient_id and existing_session.patient_id != session.patient_id:
                    session.ecg_instance_id = None
                    logging.info(f"[SessionManager] Cleared ecg_instance_id due to patient_id change from {existing_session.patient_id} to {session.patient_id}")
                
                # If existing session has patient_id and new session doesn't, keep the existing one
                if existing_session.patient_id and not session.patient_id:
                    session.patient_id = existing_session.patient_id
                    logging.info(f"[SessionManager] Kept existing patient_id: {session.patient_id} for session {session.session_id}")
                
                # If existing session has ecg_instance_id and new session doesn't, keep the existing one
                if existing_session.ecg_instance_id and not session.ecg_instance_id:
                    # Only keep if patient_id hasn't changed
                    if existing_session.patient_id == session.patient_id:
                        session.ecg_instance_id = existing_session.ecg_instance_id
                        logging.info(f"[SessionManager] Kept existing ecg_instance_id: {session.ecg_instance_id} for session {session.session_id}")
            
            # Update access time
            session.last_access = datetime.now()
            
            # Save session
            self._sessions[session.session_id] = session
            logging.info(f"[SessionManager] Saved session {session.session_id} with patient_id: {session.patient_id}, ecg_instance_id: {session.ecg_instance_id}")
        except Exception as e:
            logging.error(f"[SessionManager] Error saving session: {e}")
    
    async def delete_session(self, session_id: str):
        if session_id in self._sessions:
            self._sessions.pop(session_id)
            logging.info(f"[SessionManager] Deleted session {session_id}")
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        now = datetime.now()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.last_access > timedelta(hours=24)
        ]
        for sid in expired:
            self.sessions.pop(sid)
            logging.info(f"[SessionManager] Cleaned up expired session {sid}")
