import pydicom
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
from app.model.ecg_file import EcgFile

class ECGProcessor:
    @staticmethod
    def load_dicom(file_path: str) -> EcgFile:
        """加载DICOM格式的ECG数据"""
        ds = pydicom.dcmread(file_path)
        
        # 提取波形数据
        waveform_sequence = ds.WaveformSequence[0]
        channel_definitions = waveform_sequence.ChannelDefinitionSequence
        waveform_data = waveform_sequence.WaveformData
        
        # 解析采样率
        sampling_rate = float(waveform_sequence.SamplingFrequency)
        
        # 解析导联数据
        leads = {}
        for i, channel in enumerate(channel_definitions):
            lead_name = channel.ChannelSourceSequence[0].CodeMeaning
            start_idx = i * len(waveform_data) // len(channel_definitions)
            end_idx = (i + 1) * len(waveform_data) // len(channel_definitions)
            leads[lead_name] = list(waveform_data[start_idx:end_idx])
        
        return EcgFile(
            patient_id=ds.PatientID,
            dicom_uid=ds.SOPInstanceUID,
            leads=leads,
            sampling_rate=sampling_rate
        )
    
    @staticmethod
    def load_csv(file_path: str) -> EcgFile:
        """加载CSV格式的ECG数据"""
        df = pd.read_csv(file_path)
        
        # 假设CSV文件格式为:
        # time, lead_i, lead_ii, lead_iii, ...
        leads = {}
        for column in df.columns:
            if column.lower() != "time":
                leads[column] = df[column].tolist()
        
        # 计算采样率
        time_diff = df["time"].iloc[1] - df["time"].iloc[0]
        sampling_rate = 1.0 / time_diff if time_diff > 0 else 1000  # 默认1kHz
        
        return EcgFile(
            patient_id=os.path.splitext(os.path.basename(file_path))[0],
            dicom_uid=f"CSV_{os.path.basename(file_path)}",
            leads=leads,
            sampling_rate=sampling_rate
        )
    
    @staticmethod
    def prepare_plot_data(ecg_data: EcgFile, duration: float = 10.0) -> Dict:
        """准备用于前端显示的ECG数据"""
        plot_data = {}
        samples_per_lead = int(duration * ecg_data.sampling_rate)
        
        for lead_name, values in ecg_data.leads.items():
            # 确保数据长度足够
            if len(values) < samples_per_lead:
                continue
                
            # 生成时间轴
            time_points = np.linspace(0, duration, samples_per_lead)
            
            plot_data[lead_name] = {
                "time": time_points.tolist(),
                "values": values[:samples_per_lead]
            }
        
        return {
            "patient_id": str(ecg_data.patient_id),
            "leads": plot_data,
            "sampling_rate": ecg_data.sampling_rate,
            "duration": duration
        }
    
    @staticmethod
    def analyze_waveform(ecg_data: Dict) -> Dict:
        """分析ECG波形数据并返回诊断结果"""
        try:
            # 提取关键参数
            sampling_rate = ecg_data.get('sampling_rate', 500)  # 默认500Hz
            waveform_data = ecg_data.get('waveform_data', [])
            leads = ecg_data.get('leads', [])
            
            if not waveform_data or not leads:
                return {"error": "Invalid ECG data format"}
            
            # 基本的ECG分析
            # 这里是一个简单的示例，实际应用中需要更复杂的分析算法
            analysis = {
                "heart_rate": "72-80 bpm",  # 示例值
                "rhythm": "Normal sinus rhythm",  # 示例值
                "abnormalities": [],
                "conclusion": "Normal ECG pattern"
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error analyzing ECG: {str(e)}"}
