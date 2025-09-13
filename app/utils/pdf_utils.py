import os
import uuid
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
import logging
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directory for storing generated PDF reports
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports")

# Create reports directory if it doesn't exist
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_pdf_report(patient_info, ecg_data, diagnosis_data, ecg_image_path=None):
    """
    Generate a PDF report for a patient's ECG data and diagnosis
    
    Args:
        patient_info (dict): Patient information
        ecg_data (dict): ECG study data
        diagnosis_data (dict): Diagnosis information
        ecg_image_path (str, optional): Path to ECG waveform image
        
    Returns:
        tuple: (file_path, file_name) - Path to the generated PDF file and the filename
    """
    try:
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_id = patient_info.get('patient_id', 'unknown')
        file_name = f"ecg_report_{patient_id}_{timestamp}.pdf"
        file_path = os.path.join(REPORTS_DIR, file_name)
        
        # Create a buffer for the PDF
        buffer = BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=16,
            alignment=1,  # Center alignment
            spaceAfter=0.3*inch
        )
        
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=0.2*inch,
            spaceAfter=0.1*inch
        )
        
        normal_style = styles['Normal']
        
        # Content elements
        elements = []
        
        # Title
        elements.append(Paragraph("ECG Diagnostic Report", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Report generation info
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Patient Information section
        elements.append(Paragraph("Patient Information", heading_style))
        
        patient_data = [
            ["Name:", patient_info.get('name', 'Not Available')],
            ["Patient ID:", patient_info.get('patient_id', 'Not Available')],
            ["Gender:", patient_info.get('gender', 'Not Available')],
            ["Birth Date:", patient_info.get('birth_date', 'Not Available')]
        ]
        
        patient_table = Table(patient_data, colWidths=[1.5*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6)
        ]))
        elements.append(patient_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # ECG Study Information
        elements.append(Paragraph("ECG Study Information", heading_style))
        
        # Format date and time
        date_str = "Unknown"
        time_str = "Unknown"
        
        if 'study_date' in ecg_data:
            date = ecg_data['study_date']
            if len(date) >= 8:
                date_str = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                
        if 'study_time' in ecg_data:
            time = ecg_data['study_time']
            if len(time) >= 4:
                time_str = f"{time[:2]}:{time[2:4]}"
        
        ecg_info_data = [
            ["Study Date:", date_str],
            ["Study Time:", time_str]
        ]
        
        # Add waveform info if available
        waveform = ecg_data.get("waveform", {})
        if waveform:
            if 'sampling_rate' in waveform:
                ecg_info_data.append(["Sampling Rate:", f"{waveform.get('sampling_rate')} Hz"])
            if 'duration' in waveform:
                ecg_info_data.append(["Duration:", f"{waveform.get('duration')} seconds"])
            if 'leads' in waveform:
                ecg_info_data.append(["Leads:", ", ".join(waveform.get('leads', []))])
        
        ecg_table = Table(ecg_info_data, colWidths=[1.5*inch, 4*inch])
        ecg_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6)
        ]))
        elements.append(ecg_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Add ECG image if available
        if ecg_image_path and os.path.exists(ecg_image_path):
            elements.append(Paragraph("ECG Waveform", heading_style))
            img = Image(ecg_image_path, width=6*inch, height=3*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
        
        # Diagnostic Results
        if diagnosis_data:
            elements.append(Paragraph("Diagnostic Results", heading_style))
            
            diagnosis_data_table = [
                ["Interpretation:", diagnosis_data.get('interpretation', 'Not Available')]
            ]
            
            diag_table = Table(diagnosis_data_table, colWidths=[1.5*inch, 4*inch])
            diag_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 6)
            ]))
            elements.append(diag_table)
            elements.append(Spacer(1, 0.2*inch))
            
            # Add probabilities if available
            probabilities = diagnosis_data.get('probabilities', {})
            if probabilities:
                elements.append(Paragraph("Probability of Conditions", heading_style))
                
                prob_data = []
                for condition, prob in probabilities.items():
                    try:
                        prob_value = float(prob) if isinstance(prob, str) else prob
                        prob_data.append([condition, f"{prob_value:.1%}"])
                    except (ValueError, TypeError):
                        prob_data.append([condition, str(prob)])
                
                if prob_data:
                    prob_table = Table(prob_data, colWidths=[3*inch, 2.5*inch])
                    prob_table.setStyle(TableStyle([
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('PADDING', (0, 0), (-1, -1), 6)
                    ]))
                    elements.append(prob_table)
        
        # Build the PDF
        doc.build(elements)
        
        # Get the PDF from the buffer
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Write to file
        with open(file_path, 'wb') as f:
            f.write(pdf_data)
        
        return file_path, file_name
        
    except Exception as e:
        logging.error(f"Error generating PDF report: {str(e)}", exc_info=True)
        return None, None

def get_report_download_url(file_name):
    """
    Get the download URL for a report
    
    Args:
        file_name (str): Name of the report file
        
    Returns:
        str: URL for downloading the report
    """
    # Get the base URL from environment variables, default to empty string if not available
    base_url = os.getenv("AIAGENT_URL", "")
    
    # If base_url is available, use it, otherwise use relative URL
    if base_url:
        return f"{base_url}/api/reports/download/{file_name}"
    else:
        return f"/api/reports/download/{file_name}"
