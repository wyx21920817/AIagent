from mongoengine import Document, StringField, DateTimeField, DictField
from datetime import datetime

class Patient(Document):
    patient_id = StringField(required=True, unique=True)
    name = StringField(required=True)
    gender = StringField(required=False)
    birth_date = DateTimeField(required=False)
    contact_info = DictField(default={})
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'patients',
        'indexes': ['patient_id']
    }

    def save(self, *args, **kwargs):
        if not self.created_at:
            self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        return super(Patient, self).save(*args, **kwargs)
