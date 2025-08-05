from mongoengine import Document, ReferenceField, DateTimeField
from datetime import datetime
from .patient import Patient
from .user import User

class DoctorPatient(Document):
    doctor = ReferenceField(User, required=True)
    patient = ReferenceField(Patient, required=True)
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'doctor_patients',
        'indexes': [
            {'fields': ['doctor', 'patient'], 'unique': True}
        ]
    }

    def save(self, *args, **kwargs):
        if not self.created_at:
            self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        return super(DoctorPatient, self).save(*args, **kwargs)
