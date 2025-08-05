from mongoengine import Document, StringField, DateTimeField, EnumField
from datetime import datetime
from enum import Enum
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserRole(Enum):
    DOCTOR = "doctor"
    ADMIN = "admin"

class User(Document):
    username = StringField(required=True, unique=True)
    hashed_password = StringField(required=True)
    role = EnumField(UserRole, required=True)
    email = StringField(required=True, unique=True)
    mfa_code = StringField()
    mfa_expires_at = DateTimeField()
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'users',
        'indexes': ['username', 'email']
    }

    def set_username(self, username: str):
        """Hash and set the set_username"""
        self.username = username

    def set_password(self, password: str):
        """Hash and set the password"""
        self.hashed_password = pwd_context.hash(password)

    def verify_password(self, password: str) -> bool:
        """Verify the password"""
        return pwd_context.verify(password, self.hashed_password)

    def save(self, *args, **kwargs):
        if not self.created_at:
            self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        return super(User, self).save(*args, **kwargs)
