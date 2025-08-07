from typing import List, Dict, Optional
from .models import Tool
from app.model.patient import Patient
from app.model.ecg_file import EcgFile
from app.core.database import Database
from app.services.ecg import ECGProcessor
from app.utils.orthanc import get_patient_instances, get_instance_waveform
import os
import requests
from fastapi import HTTPException

class ECGTools:
    def __init__(self):
        self.db = Database()
        self.ecg_processor = ECGProcessor()
        self.orthanc_url = os.getenv('ORTHANC_SERVER_URL', 'http://127.0.0.1:8042')
        self.auth_url = os.getenv('AUTH_SERVER_URL', 'http://54.79.181.49:8001')
    
    async def get_temp_token(self) -> str:
        """Get temporary authentication token"""
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        try:
            response = requests.get(
                f'{self.auth_url}/api/auth/temp-token',
                headers=headers,
                verify=False,
                timeout=5.0
            )
            response.raise_for_status()
            data = response.json()
            temp_token = data.get('token')
            if not temp_token:
                raise ValueError("Token not found in response")
   
            return temp_token
        except requests.Timeout:

            raise HTTPException(status_code=504, detail="Timeout getting temporary token, please try again later")
        except requests.HTTPError as e:

            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Failed to get temporary token: {e.response.text}"
            )
        except requests.RequestException as e:

            raise HTTPException(status_code=503, detail="Cannot connect to authentication server, please check network connection")
    
    async def get_patient_data(self, patient_id: str, token: str = None) -> Dict:
        """Get patient data and diagnostic results"""
        
        # If no token provided, get temporary token
        if not token:
            token = await self.get_temp_token()
            
        # Get all patient instances
        instances = get_patient_instances(patient_id, token)
        if not instances:
            return {"error": "No instances found for patient"}
            
        # Filter ECG instances and sort by time
        ecg_instances = []
        patient_info = None
        
        for instance in instances:
            # Get patient information
            if not patient_info and 'MainDicomTags' in instance:
                patient_info = {
                    'PatientID': instance['MainDicomTags'].get('PatientID', ''),
                    'PatientName': instance['MainDicomTags'].get('PatientName', ''),
                    'PatientBirthDate': instance['MainDicomTags'].get('PatientBirthDate', ''),
                    'PatientSex': instance['MainDicomTags'].get('PatientSex', '')
                }
                
            # Check if this is an ECG instance
            tags = instance.get('MainDicomTags', {})
            modality = tags.get('Modality', '')
            if modality == 'ECG':
                # Get instance timestamp
                study_date = tags.get('StudyDate', '')
                study_time = tags.get('StudyTime', '')
                if study_date and study_time:
                    # Combine date and time
                    timestamp = f"{study_date}{study_time}"
                    ecg_instances.append((timestamp, instance))
        
        if not patient_info:
            return {"error": "Patient information not found"}
        
        result = {
            "patient_id": patient_id,
            "patient_info": patient_info,
            "has_ecg": False
        }
        
        if ecg_instances:
            # Sort by timestamp and get the latest ECG instance
            ecg_instances.sort(key=lambda x: x[0], reverse=True)
            latest_instance = ecg_instances[0][1]
            
            # Get ECG waveform data
            ecg_data = get_instance_waveform(latest_instance['ID'], token)
            if ecg_data:
                result.update({
                    "has_ecg": True,
                    "ecg_data": ecg_data,
                    "latest_ecg": {
                        "instance_id": latest_instance['ID'],
                        "study_date": latest_instance['MainDicomTags'].get('StudyDate', ''),
                        "study_time": latest_instance['MainDicomTags'].get('StudyTime', '')
                    }
                })
        
        return result
    
    async def get_ecg_plot_data(self, patient_id: str, token: str = None, duration: float = 10.0) -> Dict:
        """Get ECG waveform data for plotting"""
        # If no token provided, get temporary token
        if not token:
            token = await self.get_temp_token()
            
        # Get all patient instances
        instances = get_patient_instances(patient_id, token)
        if not instances:
            return {"error": "No instances found for patient"}
            
        # Filter ECG instances and sort by time
        ecg_instances = []
        for instance in instances:
            # Check if this is an ECG instance
            tags = instance.get('MainDicomTags', {})
            modality = tags.get('Modality', '')
            if modality == 'ECG':
                # Get instance timestamp
                study_date = tags.get('StudyDate', '')
                study_time = tags.get('StudyTime', '')
                if study_date and study_time:
                    # Combine date and time
                    timestamp = f"{study_date}{study_time}"
                    ecg_instances.append((timestamp, instance))
                    
        if not ecg_instances:
            return {"error": "No ECG data found for patient"}
            
        # Sort by timestamp and get the latest ECG instance
        ecg_instances.sort(key=lambda x: x[0], reverse=True)
        latest_instance = ecg_instances[0][1]
        
        # Get ECG waveform data
        ecg_data = get_instance_waveform(latest_instance['ID'], token)
        if not ecg_data:
            return {"error": "No waveform data found in ECG instance"}
            
        # Process data for plotting
        plot_data = self.ecg_processor.prepare_plot_data(ecg_data, duration)
        
        # Add additional information
        plot_data.update({
            "patient_id": patient_id,
            "instance_id": latest_instance['ID'],
            "sampling_rate": ecg_data.get('sampling_rate', 0),
            "duration": duration,
            "study_date": latest_instance['MainDicomTags'].get('StudyDate', ''),
            "study_time": latest_instance['MainDicomTags'].get('StudyTime', '')
        })
        
        return plot_data
    
    async def analyze_ecg(self, patient_id: str, token: str = None) -> Dict:
        """Analyze ECG data and return diagnostic results"""
        
        # If no token provided, get temporary token
        if not token:
            token = await self.get_temp_token()
            
        # Get all patient instances
        instances = get_patient_instances(patient_id, token)
        if not instances:
            return {"error": "No instances found for patient"}
            
        # Filter ECG instances and sort by time
        ecg_instances = []
        for instance in instances:
            # Check if this is an ECG instance
            tags = instance.get('MainDicomTags', {})
            modality = tags.get('Modality', '')
            if modality == 'ECG':
                # Get instance timestamp
                study_date = tags.get('StudyDate', '')
                study_time = tags.get('StudyTime', '')
                if study_date and study_time:
                    # Combine date and time
                    timestamp = f"{study_date}{study_time}"
                    ecg_instances.append((timestamp, instance))
                    
        if not ecg_instances:
            return {"error": "No ECG data found for patient"}
            
        # Sort by timestamp and get the latest ECG instance
        ecg_instances.sort(key=lambda x: x[0], reverse=True)
        latest_instance = ecg_instances[0][1]
        
        # Get ECG waveform data
        ecg_data = get_instance_waveform(latest_instance['ID'], token)
        if not ecg_data:
            return {"error": "No waveform data found in ECG instance"}
            
        # Analyze ECG data
        diagnosis = self.ecg_processor.analyze_waveform(ecg_data)
        if "error" in diagnosis:
            return {"error": diagnosis["error"]}
            
        # Add patient ID and time information to diagnostic results
        diagnosis["patient_id"] = patient_id
        diagnosis["study_date"] = latest_instance['MainDicomTags'].get('StudyDate', '')
        diagnosis["study_time"] = latest_instance['MainDicomTags'].get('StudyTime', '')
        
        return {
            "diagnosis": diagnosis,
            "waveform_info": {
                "sampling_rate": ecg_data['sampling_rate'],
                "duration": ecg_data['duration'],
                "leads": ecg_data['leads']
            }
        }


# Instantiate tools class
ecg_tools = ECGTools()

async def get_patient_data(patient_id: str) -> Dict:
    return await ecg_tools.get_patient_data(patient_id)

async def get_ecg_plot(patient_id: str, duration: float = 10.0) -> Dict:
    return await ecg_tools.get_ecg_plot_data(patient_id, duration)

async def analyze_ecg(patient_id: str) -> Dict:
    return await ecg_tools.analyze_ecg(patient_id)

def search_medical_knowledge(query: str, type: str = None) -> Dict:
    """Search medical knowledge base"""
    # Implement actual knowledge base search here
    return {
        "results": [
            {
                "title": "Related Medical Information",
                "content": "Medical content will be returned here",
                "confidence": 0.95
            }
        ]
    }

def recommend_treatment(condition: str, patient_info: Dict = None) -> Dict:
    """Recommend treatment plan"""
    # Implement actual treatment plan recommendation here
    return {
        "treatments": [
            {
                "name": "Recommended Treatment Plan",
                "description": "Detailed description of the treatment plan",
                "confidence": 0.9
            }
        ]
    }
