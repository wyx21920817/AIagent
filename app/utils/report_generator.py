import os
import io
import logging
import requests
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER

logger = logging.getLogger(__name__)

# Directory for storing ECG reports
ECG_REPORTS_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports", "ecg"))
ECG_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def download_image(image_url, instance_id):
    """
    Download an image from the AI server and save it locally.
    
    Args:
        image_url: The URL of the image on the AI server
        instance_id: The instance ID for creating a local directory
        
    Returns:
        Path to the downloaded image or None if download failed
    """
    import os
    try:
        # Create directory for this instance's images if it doesn't exist
        image_dir = ECG_REPORTS_DIR / instance_id
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the image filename from the URL
        image_filename = os.path.basename(image_url)
        local_image_path = image_dir / image_filename
        
        # Always download a fresh copy of the image
        logger.info(f"Will download a fresh copy of the image to {local_image_path}")
        
        # Get the AI server URL from environment
        ai_server_url = os.getenv('AISERVER_URI')
        
        # Construct the full URL to the image on the AI server
        # Check if image_url already has http:// or https:// prefix
        if image_url.startswith(('http://', 'https://')):
            full_url = image_url
        # Check if image_url is an absolute path starting with /static/
        elif image_url.startswith('/static/'):
            # For static files, use the local path instead of making a request
            # Extract the filename from the path
            filename = os.path.basename(image_url)
            # Create a local path for the image
            local_image_path = image_dir / filename
            # If the file doesn't exist locally, we'll need to download it from the AI server
            full_url = f"{ai_server_url}{image_url}"
        # Check if image_url is an absolute path starting with /
        elif image_url.startswith('/'):
            full_url = f"{ai_server_url}{image_url}"
        else:
            # If it's a relative path, add a / before appending to the base URL
            full_url = f"{ai_server_url}/{image_url}"
        
        logger.info(f"Downloading image from: {full_url}")
        logger.info(f"AI server URL: {ai_server_url}")
        logger.info(f"Image URL: {image_url}")
        logger.info(f"Instance ID: {instance_id}")
        logger.info(f"Local image path: {local_image_path}")
        
        # Download the image
        response = requests.get(full_url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Save the image locally
        with open(local_image_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return local_image_path
    
    except Exception as e:
        logger.error(f"Error downloading image {image_url}: {str(e)}")
        return None


def generate_ecg_report_pdf(interpretation_data, output_path, instance_id=None):
    """
    Generate a PDF report from the ECG interpretation data.
    
    Args:
        interpretation_data: The interpretation data from the AI server
        output_path: The path to save the PDF file
        instance_id: The instance ID for downloading images
    """
    # If instance_id is not provided, try to get it from the interpretation data
    if instance_id is None:
        instance_id = interpretation_data.get("instance_id", "unknown")
    # Create a buffer for the PDF
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get the styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=12
    )
    
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=6
    )
    
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6
    )
    
    # Create the content elements
    elements = []
    
    # Add the title
    elements.append(Paragraph(f"ECG Interpretation Report", title_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add the instance ID
    instance_id = interpretation_data.get("instance_id", "Unknown")
    elements.append(Paragraph(f"Instance ID: {instance_id}", normal_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Add the date
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Report Date: {current_date}", normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add the abnormalities section
    abnormalities = interpretation_data.get("abnormalities", [])
    if abnormalities:
        elements.append(Paragraph("Detected Abnormalities:", heading_style))
        for abnormality in abnormalities:
            elements.append(Paragraph(f"• {abnormality}", normal_style))
        elements.append(Spacer(1, 0.25*inch))
    
    # Add the results section
    results = interpretation_data.get("results", [])
    if results:
        elements.append(Paragraph("Analysis Results:", heading_style))
        elements.append(Spacer(1, 0.1*inch))
        
        for result in results:
            status = result.get("status", "Unknown")
            abnormality = result.get("abnormality", "Unknown")
            
            # Add the result header
            elements.append(Paragraph(f"{abnormality} Analysis (Status: {status})", heading_style))
            
            if status == "ok":
                # Add the report details
                report = result.get("report", {})
                if report:
                    # Add the report title
                    title = report.get("title", "")
                    if title:
                        elements.append(Paragraph(f"Title: {title}", normal_style))
                    
                    # Add the report text
                    report_text = report.get("report_text", "")
                    if report_text:
                        # Split the report text by lines and add each line
                        for line in report_text.split("\n"):
                            elements.append(Paragraph(line, normal_style))
                    
                    # Add images if available
                    image_paths = report.get("image_paths", {})
                    if image_paths:
                        elements.append(Spacer(1, 0.1*inch))
                        elements.append(Paragraph("Images:", heading_style))
                        
                        # Download and add each image
                        for image_name, image_url in image_paths.items():
                            # Download the image
                            local_image_path = download_image(image_url, instance_id)
                            
                            if local_image_path and local_image_path.exists():
                                try:
                                    # Add image caption
                                    elements.append(Paragraph(f"{image_name}:", normal_style))
                                    
                                    # Add the image with a reasonable width
                                    img = Image(str(local_image_path), width=5*inch, height=3*inch)
                                    elements.append(img)
                                    elements.append(Spacer(1, 0.1*inch))
                                except Exception as e:
                                    logger.error(f"Error adding image to PDF: {str(e)}")
                                    elements.append(Paragraph(f"Error loading image: {image_name}", normal_style))
                    
                    # Add metrics if available
                    metrics = report.get("metrics", {})
                    if metrics:
                        elements.append(Paragraph("Metrics:", heading_style))
                        
                        # Create a table for the metrics
                        metrics_data = []
                        for key, value in metrics.items():
                            if isinstance(value, dict):
                                # Handle nested dictionaries
                                for sub_key, sub_value in value.items():
                                    metrics_data.append([f"{key}.{sub_key}", str(sub_value)])
                            else:
                                metrics_data.append([key, str(value)])
                        
                        if metrics_data:
                            # Create the table
                            metrics_table = Table(metrics_data, colWidths=[2*inch, 3*inch])
                            metrics_table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                                ('GRID', (0, 0), (-1, -1), 1, colors.black)
                            ]))
                            elements.append(metrics_table)
                    
                    # Add validation information if available
                    validation = report.get("validation", {})
                    if validation:
                        elements.append(Spacer(1, 0.1*inch))
                        elements.append(Paragraph("Validation:", heading_style))
                        
                        # Add validation status
                        validation_status = validation.get("status", "Unknown")
                        elements.append(Paragraph(f"Status: {validation_status}", normal_style))
                        
                        # Add validation reasons if available
                        reasons = validation.get("reasons", [])
                        if reasons:
                            elements.append(Paragraph("Reasons:", normal_style))
                            for reason in reasons:
                                elements.append(Paragraph(f"• {reason}", normal_style))
            else:
                # Add the reason for skipping
                reason = result.get("reason", "Unknown")
                elements.append(Paragraph(f"Reason: {reason}", normal_style))
            
            elements.append(Spacer(1, 0.25*inch))
    
    # Add the message
    message = interpretation_data.get("message", "")
    if message:
        elements.append(Paragraph("Message:", heading_style))
        elements.append(Paragraph(message, normal_style))
    
    # Build the PDF
    doc.build(elements)
    
    # Get the PDF data from the buffer
    pdf_data = buffer.getvalue()
    buffer.close()
    
    # Write the PDF to the output path
    with open(output_path, "wb") as f:
        f.write(pdf_data)
    
    return output_path
