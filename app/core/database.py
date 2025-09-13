from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, List, Dict
from datetime import datetime
from dotenv import load_dotenv
import os
import logging
from mongoengine import connect
from app.model.ecg_file import EcgFile
from app.model.patient import Patient

# Load environment variables
load_dotenv()

# Initialize mongoengine connection
mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')

print('mongodb_uri:-------------------'+mongodb_uri)
connect('aiclinic', host=mongodb_uri)

class Database:
    def __init__(self):
        # Initialize MongoDB client with URI from .env
        mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
        self.client = AsyncIOMotorClient(mongodb_uri)
        self.db = self.client.aiclinic
    
    async def get_patient(self, patient_id: str) -> Optional[Patient]:
        """Get patient by ID (either MongoDB _id or patient_id field)"""
        try:
            # First try to query by MongoDB _id
            from bson import ObjectId
            try:
                if len(patient_id) == 24:  # MongoDB ObjectId is 24 characters
                    doc = await self.db.patients.find_one({"_id": ObjectId(patient_id)})
                    if doc:
                        doc["id"] = str(doc.pop("_id"))
                        return Patient(**doc)
            except Exception as e:
                logging.debug(f"Not a valid ObjectId: {patient_id}, trying patient_id field")
            
            # If not ObjectId or not found, try to query by patient_id field
            doc = await self.db.patients.find_one({"patient_id": patient_id})
            if doc:
                doc["id"] = str(doc.pop("_id"))
                return Patient(**doc)
                
            return None
        except Exception as e:
            logging.error(f"Error getting patient: {e}")
            return None
    
    async def find_patient_by_name(self, name: str) -> Optional[Patient]:
        """Find patient by name using simple fuzzy matching"""
        try:
            doc = await self.db.patients.find_one({"name": {"$regex": name, "$options": "i"}})
            if not doc:
                return None
            return Patient(
                id=str(doc["_id"]),
                patient_id=doc.get("patient_id"),
                name=doc.get("name"),
                gender=doc.get("gender"),
                birth_date=doc.get("birth_date"),
                contact_info=doc.get("contact_info"),
            )
        except Exception as e:
            logging.error(f"Error finding patient by name: {e}")
            return None
    
    async def get_ecg_data(self, ecg_id: str) -> Optional[EcgFile]:
        """Get ECG data by id"""
        try:
            result = await self.db.ecg_files.find_one({"_id": str(ecg_id)})
            if result:
                return EcgFile(**result)
            return None
        except Exception as e:
            logging.error(f"Error getting ECG data: {e}")
            return None
    
    async def get_diagnosis(self, patient_id: str) -> Optional[Dict]:
        """Get latest diagnosis result from ECG file"""
        try:
            ecg = await self.get_latest_ecg(patient_id)
            if ecg and ecg.diagnosis_result:
                return ecg.diagnosis_result
            return None
        except Exception as e:
            logging.error(f"Error getting diagnosis: {e}")
            return None
    
    async def list_patients(self) -> List[Patient]:
        """Get list of all patients"""
        cursor = self.db.patients.find({})
        patients = []
        async for doc in cursor:
            # Convert _id to id for Pydantic model
            doc["id"] = str(doc.pop("_id"))
            patients.append(Patient(**doc))
        return patients
    
    async def save_patient(self, patient: Patient) -> bool:
        """Save patient information"""
        try:
            # Convert the model to dict and ensure _id is set
            data = patient.dict()
            data["_id"] = str(patient.id)  # Ensure ID is string
            
            await self.db.patients.update_one(
                {"_id": data["_id"]},
                {"$set": data},
                upsert=True
            )
            return True
        except Exception as e:
            logging.error(f"Error saving patient: {e}")
            return False
    
    async def save_ecg_data(self, ecg_data: EcgFile) -> bool:
        """Save ECG data"""
        try:
            await ecg_data.save()
            return True
        except Exception as e:
            logging.error(f"Error saving ECG data: {e}")
            return False
    
    async def get_patient_diagnoses(self, patient_id: str) -> List[Dict]:
        """Get all diagnoses for a patient from ECG files"""
        try:
            # First get patient record to confirm existence
            doc = await self.db.patients.find_one({"patient_id": patient_id})
            if not doc:
                logging.warning(f"[get_patient_diagnoses] Patient not found: {patient_id}")
                return []
            
            patient_object_id = doc["_id"]
            logging.info(f"[get_patient_diagnoses] Found patient with _id: {patient_object_id}")
            
            # Query ECG files using patient's _id, sort by created_at in descending order to get the newest first
            cursor = self.db.ecg_files.find({"patient_id": patient_object_id}).sort("created_at", -1)
            diagnoses = []
            async for doc in cursor:
                if not doc.get("diagnosis_result"):
                    continue
                    
                diagnosis = doc["diagnosis_result"]
                # Ensure diagnosis result includes timestamp
                if "diagnosis_time" not in diagnosis:
                    diagnosis["diagnosis_time"] = doc.get("created_at", datetime.utcnow())
                if "created_at" not in diagnosis:
                    diagnosis["created_at"] = doc.get("created_at", datetime.utcnow())
                    
                # Add some metadata
                diagnosis["ecg_id"] = str(doc["_id"])
                diagnosis["patient_id"] = patient_id  # Save original patient_id for reference
                    
                diagnoses.append(diagnosis)
                
            logging.info(f"[get_patient_diagnoses] Found {len(diagnoses)} diagnoses for patient {patient_id}")
            return diagnoses
            
        except Exception as e:
            logging.error(f"[get_patient_diagnoses] Error getting patient diagnoses: {e}", exc_info=True)
            return []

    async def get_latest_ecg(self, patient_id: str) -> Optional[EcgFile]:
        """Get the most recent ECG data for a patient"""
        try:
            # Sort by created_at in descending order and get the first one
            result = await self.db.ecg_files.find_one(
                {"patient_id": patient_id},
                sort=[("created_at", -1)]
            )
            if result:
                return EcgFile(**result)
            return None
        except Exception as e:
            logging.error(f"Error getting latest ECG: {e}")
            return None

    async def save_diagnosis(self, ecg_id: str, diagnosis: Dict) -> bool:
        """Save diagnosis result to ECG file"""
        try:
            await self.db.ecg_files.update_one(
                {"_id": ecg_id},
                {"$set": {"diagnosis_result": diagnosis}}
            )
            return True
        except Exception as e:
            logging.error(f"Error saving diagnosis: {e}")
            return False
