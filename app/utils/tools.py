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
        self.auth_url = os.getenv('AUTH_SERVER_URL', 'http://52.63.231.27:8001')
    
    async def get_temp_token(self) -> str:
        """获取临时令牌"""
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

            raise HTTPException(status_code=504, detail="获取临时令牌超时，请稍后重试")
        except requests.HTTPError as e:

            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"获取临时令牌失败: {e.response.text}"
            )
        except requests.RequestException as e:

            raise HTTPException(status_code=503, detail="无法连接到认证服务器，请检查网络连接")
    
    async def get_patient_data(self, patient_id: str, token: str = None) -> Dict:
        """获取患者数据和诊断结果"""
        
        # 如果没有提供token，获取临时令牌
        if not token:
            token = await self.get_temp_token()
            
        # 获取患者的所有实例
        instances = get_patient_instances(patient_id, token)
        if not instances:
            return {"error": "No instances found for patient"}
            
        # 筛选ECG实例并按时间排序
        ecg_instances = []
        patient_info = None
        
        for instance in instances:
            # 获取患者信息
            if not patient_info and 'MainDicomTags' in instance:
                patient_info = {
                    'PatientID': instance['MainDicomTags'].get('PatientID', ''),
                    'PatientName': instance['MainDicomTags'].get('PatientName', ''),
                    'PatientBirthDate': instance['MainDicomTags'].get('PatientBirthDate', ''),
                    'PatientSex': instance['MainDicomTags'].get('PatientSex', '')
                }
                
            # 检查是否是ECG实例
            tags = instance.get('MainDicomTags', {})
            modality = tags.get('Modality', '')
            if modality == 'ECG':
                # 获取实例的时间戳
                study_date = tags.get('StudyDate', '')
                study_time = tags.get('StudyTime', '')
                if study_date and study_time:
                    # 组合日期和时间
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
            # 按时间戳排序，获取最新的ECG实例
            ecg_instances.sort(key=lambda x: x[0], reverse=True)
            latest_instance = ecg_instances[0][1]
            
            # 获取ECG波形数据
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
        """获取ECG波形数据用于绘图"""
        # 如果没有提供token，获取临时令牌
        if not token:
            token = await self.get_temp_token()
            
        # 获取患者的所有实例
        instances = get_patient_instances(patient_id, token)
        if not instances:
            return {"error": "No instances found for patient"}
            
        # 筛选ECG实例并按时间排序
        ecg_instances = []
        for instance in instances:
            # 检查是否是ECG实例
            tags = instance.get('MainDicomTags', {})
            modality = tags.get('Modality', '')
            if modality == 'ECG':
                # 获取实例的时间戳
                study_date = tags.get('StudyDate', '')
                study_time = tags.get('StudyTime', '')
                if study_date and study_time:
                    # 组合日期和时间
                    timestamp = f"{study_date}{study_time}"
                    ecg_instances.append((timestamp, instance))
                    
        if not ecg_instances:
            return {"error": "No ECG data found for patient"}
            
        # 按时间戳排序，获取最新的ECG实例
        ecg_instances.sort(key=lambda x: x[0], reverse=True)
        latest_instance = ecg_instances[0][1]
        
        # 获取ECG波形数据
        ecg_data = get_instance_waveform(latest_instance['ID'], token)
        if not ecg_data:
            return {"error": "No waveform data found in ECG instance"}
            
        # 处理数据用于绘图
        plot_data = self.ecg_processor.prepare_plot_data(ecg_data, duration)
        
        # 添加额外信息
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
        """分析ECG数据并返回诊断结果"""
        
        # 如果没有提供token，获取临时令牌
        if not token:
            token = await self.get_temp_token()
            
        # 获取患者的所有实例
        instances = get_patient_instances(patient_id, token)
        if not instances:
            return {"error": "No instances found for patient"}
            
        # 筛选ECG实例并按时间排序
        ecg_instances = []
        for instance in instances:
            # 检查是否是ECG实例
            tags = instance.get('MainDicomTags', {})
            modality = tags.get('Modality', '')
            if modality == 'ECG':
                # 获取实例的时间戳
                study_date = tags.get('StudyDate', '')
                study_time = tags.get('StudyTime', '')
                if study_date and study_time:
                    # 组合日期和时间
                    timestamp = f"{study_date}{study_time}"
                    ecg_instances.append((timestamp, instance))
                    
        if not ecg_instances:
            return {"error": "No ECG data found for patient"}
            
        # 按时间戳排序，获取最新的ECG实例
        ecg_instances.sort(key=lambda x: x[0], reverse=True)
        latest_instance = ecg_instances[0][1]
        
        # 获取ECG波形数据
        ecg_data = get_instance_waveform(latest_instance['ID'], token)
        if not ecg_data:
            return {"error": "No waveform data found in ECG instance"}
            
        # 分析ECG数据
        diagnosis = self.ecg_processor.analyze_waveform(ecg_data)
        if "error" in diagnosis:
            return {"error": diagnosis["error"]}
            
        # 添加患者ID和时间信息到诊断结果中
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

def get_available_tools() -> List[Tool]:
    """Return list of available tools for the doctor agent"""
    return [
        Tool(
            name="get_patient_data",
            description="获取患者基本信息和诊断结果概要",
            parameters={
                "patient_id": "string"
            },
            required=["patient_id"]
        ),
        Tool(
            name="get_ecg_plot",
            description="获取ECG波形数据用于绘图显示",
            parameters={
                "patient_id": "string",
                "duration": "float, 显示时长(秒)"
            },
            required=["patient_id"]
        ),
        Tool(
            name="analyze_ecg",
            description="分析ECG数据并返回详细的诊断结果",
            parameters={
                "patient_id": "string"
            },
            required=["patient_id"]
        ),
        Tool(
            name="search_medical_knowledge",
            description="搜索医学知识库获取相关信息",
            parameters={
                "query": "string",
                "type": "string (condition/symptom/treatment)"
            },
            required=["query"]
        ),
        Tool(
            name="recommend_treatment",
            description="基于诊断结果推荐治疗方案",
            parameters={
                "condition": "string",
                "patient_info": "dict, 可选的患者信息"
            },
            required=["condition"]
        )
    ]

# 实例化工具类
ecg_tools = ECGTools()

async def get_patient_data(patient_id: str) -> Dict:
    return await ecg_tools.get_patient_data(patient_id)

async def get_ecg_plot(patient_id: str, duration: float = 10.0) -> Dict:
    return await ecg_tools.get_ecg_plot_data(patient_id, duration)

async def analyze_ecg(patient_id: str) -> Dict:
    return await ecg_tools.analyze_ecg(patient_id)

def search_medical_knowledge(query: str, type: str = None) -> Dict:
    """搜索医学知识库"""
    # 这里应该实现实际的知识库搜索
    return {
        "results": [
            {
                "title": "相关医学信息",
                "content": "这里将返回相关的医学内容",
                "confidence": 0.95
            }
        ]
    }

def recommend_treatment(condition: str, patient_info: Dict = None) -> Dict:
    """推荐治疗方案"""
    # 这里应该实现实际的治疗方案推荐
    return {
        "treatments": [
            {
                "name": "推荐治疗方案",
                "description": "治疗方案的详细描述",
                "confidence": 0.9
            }
        ]
    }
