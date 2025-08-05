from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    # MongoDB settings
    MONGODB_URI: str = "mongodb://localhost:27017/aiclinic"
    
    # Orthanc settings
    ORTHANC_URL: str = "http://localhost:8042"
    AUTH_SERVER_URL: str = "http://localhost:8000"
    
    # AI service settings
    AI_SERVICE_URL: str = "http://localhost:5000"
    AI_SERVICE_TOKEN: str = ""  # Set this in environment variables
    
    class Config:
        env_file = ".env"

settings = Settings()
