from mongoengine import Document, StringField, ReferenceField, ListField, DictField, DateTimeField, BooleanField
from datetime import datetime
from .user import User
from .patient import Patient

class EcgFile(Document):
    patient_id = ReferenceField(Patient, required=True)
    dicom_uid = StringField(required=True, unique=True)
    uploaded_by = ReferenceField(User, required=True)
    tags = ListField(StringField(), default=list)
    notes = StringField()
    diagnosis_result = DictField(default={})
    instanceIDs = ListField(StringField(), default=list)  # Store all related instance IDs
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    Approved_status = BooleanField(default=False)

    meta = {
        'collection': 'ecg_files',
        'indexes': ['patient_id', 'dicom_uid', 'uploaded_by']
    }

    def save(self, *args, **kwargs):
        if not self.created_at:
            self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        return super(EcgFile, self).save(*args, **kwargs)
