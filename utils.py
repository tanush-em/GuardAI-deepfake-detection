import os
import json
import base64
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import cv2
import tempfile

def generate_pdf_report(report_data, output_path=None):
    """
    Generate a comprehensive PDF report from analysis results
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        
        if output_path is None:
            output_path = f"deepfake_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        story.append(Paragraph("🛡️ GuardAI Deepfake Detection Report", title_style))
        story.append(Spacer(1, 20))
        
        # Report metadata
        story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"<b>Video File:</b> {report_data['video_info']['filename']}", styles['Normal']))
        story.append(Paragraph(f"<b>File Size:</b> {report_data['video_info']['size_mb']:.2f} MB", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Prediction results
        result = report_data['prediction']['result']
        confidence = report_data['prediction']['confidence']
        
        result_color = colors.green if result == "REAL" else colors.red
        result_text = f"<b>Detection Result:</b> <font color='{result_color}'>{result}</font>"
        story.append(Paragraph(result_text, styles['Heading2']))
        story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles['Normal']))
        story.append(Paragraph(f"<b>Entropy:</b> {report_data['prediction']['entropy']:.3f}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Analysis details
        story.append(Paragraph("<b>Analysis Details:</b>", styles['Heading2']))
        analysis_data = [
            ['Metric', 'Value'],
            ['Frames Analyzed', str(report_data['analysis']['frames_analyzed'])],
            ['Faces Detected', str(report_data['analysis']['faces_detected'])],
            ['Average Brightness', f"{report_data['analysis']['avg_brightness']:.1f}"],
            ['Average Contrast', f"{report_data['analysis']['avg_contrast']:.1f}"]
        ]
        
        analysis_table = Table(analysis_data)
        analysis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(analysis_table)
        story.append(Spacer(1, 20))
        
        # Model information
        story.append(Paragraph("<b>Model Information:</b>", styles['Heading2']))
        model_data = [
            ['Parameter', 'Value'],
            ['Model Name', report_data['model_info']['name']],
            ['Architecture', report_data['model_info']['architecture']],
            ['Sequence Length', str(report_data['model_info']['sequence_length'])],
            ['Face Detection', 'Enabled' if report_data['model_info']['face_detection'] else 'Disabled']
        ]
        
        model_table = Table(model_data)
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(model_table)
        
        # Build PDF
        doc.build(story)
        return output_path
        
    except ImportError:
        print("ReportLab not available. Install with: pip install reportlab")
        return None

def create_advanced_visualizations(prediction_results, frame_metadata):
    """
    Create advanced visualizations for the analysis results
    """
    plots = {}
    
    # Enhanced confidence gauge with multiple metrics
    fig_confidence = go.Figure()
    fig_confidence.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=prediction_results['confidence'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Detection Confidence", 'font': {'size': 20}},
        delta={'reference': 50, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig_confidence.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    plots['confidence_gauge'] = fig_confidence
    
    # Frame analysis with multiple metrics
    if frame_metadata:
        timestamps = [m['timestamp'] for m in frame_metadata]
        brightness = [m['brightness'] for m in frame_metadata]
        contrast = [m['contrast'] for m in frame_metadata]
        face_detected = [1 if m['face_detected'] else 0 for m in frame_metadata]
        
        fig_frame_analysis = go.Figure()
        
        # Brightness
        fig_frame_analysis.add_trace(go.Scatter(
            x=timestamps, y=brightness, mode='lines+markers',
            name='Brightness', line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Contrast
        fig_frame_analysis.add_trace(go.Scatter(
            x=timestamps, y=contrast, mode='lines+markers',
            name='Contrast', line=dict(color='red', width=2),
            marker=dict(size=6), yaxis='y2'
        ))
        
        # Face detection indicator
        fig_frame_analysis.add_trace(go.Scatter(
            x=timestamps, y=face_detected, mode='markers',
            name='Face Detected', marker=dict(
                size=8, 
                color='green' if any(face_detected) else 'red',
                symbol='circle'
            ),
            yaxis='y3'
        ))
        
        fig_frame_analysis.update_layout(
            title='Comprehensive Frame Analysis Over Time',
            xaxis_title='Time (seconds)',
            yaxis=dict(title='Brightness', side='left', showgrid=True),
            yaxis2=dict(title='Contrast', side='right', overlaying='y', showgrid=False),
            yaxis3=dict(title='Face Detection', side='right', overlaying='y', range=[0, 1.2], showgrid=False),
            hovermode='x unified',
            height=500,
            legend=dict(x=0.02, y=0.98)
        )
        plots['frame_analysis'] = fig_frame_analysis
    
    # Attention weights heatmap
    if 'attention_weights' in prediction_results:
        attention = prediction_results['attention_weights'][0, 0, :, :]
        fig_attention = px.imshow(
            attention, 
            title='Attention Weights Heatmap - Model Focus Areas',
            labels=dict(x="Frame Position", y="Attention Position"),
            color_continuous_scale='Viridis',
            aspect='auto'
        )
        fig_attention.update_layout(height=400)
        plots['attention_heatmap'] = fig_attention
    
    # Probability distribution
    if 'probabilities' in prediction_results:
        probs = prediction_results['probabilities'][0]
        fig_prob_dist = go.Figure(data=[
            go.Bar(x=['FAKE', 'REAL'], y=probs, 
                   marker_color=['red', 'green'],
                   text=[f'{p:.3f}' for p in probs],
                   textposition='auto')
        ])
        fig_prob_dist.update_layout(
            title='Prediction Probability Distribution',
            yaxis_title='Probability',
            height=400
        )
        plots['probability_distribution'] = fig_prob_dist
    
    return plots

def extract_video_metadata(video_path):
    """
    Extract comprehensive metadata from video file
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        metadata = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': 0,
            'codec': '',
            'bitrate': 0
        }
        
        if metadata['fps'] > 0:
            metadata['duration'] = metadata['frame_count'] / metadata['fps']
        
        # Try to get codec information
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        metadata['codec'] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        return metadata
        
    except Exception as e:
        print(f"Error extracting video metadata: {e}")
        return {}

def create_analysis_summary(reports):
    """
    Create a comprehensive summary of multiple analyses
    """
    if not reports:
        return {}
    
    summary = {
        'total_analyses': len(reports),
        'real_count': 0,
        'fake_count': 0,
        'error_count': 0,
        'avg_confidence': 0,
        'confidence_distribution': {'low': 0, 'medium': 0, 'high': 0},
        'model_performance': {},
        'time_analysis': {},
        'file_size_analysis': {}
    }
    
    confidences = []
    
    for report in reports:
        # Count predictions
        if 'prediction' in report and 'result' in report['prediction']:
            result = report['prediction']['result']
            if result == 'REAL':
                summary['real_count'] += 1
            elif result == 'FAKE':
                summary['fake_count'] += 1
            else:
                summary['error_count'] += 1
            
            # Collect confidence scores
            if 'confidence' in report['prediction']:
                conf = report['prediction']['confidence']
                confidences.append(conf)
                
                # Categorize confidence
                if conf < 50:
                    summary['confidence_distribution']['low'] += 1
                elif conf < 80:
                    summary['confidence_distribution']['medium'] += 1
                else:
                    summary['confidence_distribution']['high'] += 1
        
        # Model performance tracking
        if 'model_info' in report and 'name' in report['model_info']:
            model_name = report['model_info']['name']
            if model_name not in summary['model_performance']:
                summary['model_performance'][model_name] = {
                    'count': 0, 'avg_confidence': 0, 'real_count': 0, 'fake_count': 0
                }
            
            summary['model_performance'][model_name]['count'] += 1
            if 'confidence' in report['prediction']:
                summary['model_performance'][model_name]['avg_confidence'] += report['prediction']['confidence']
            
            if report['prediction']['result'] == 'REAL':
                summary['model_performance'][model_name]['real_count'] += 1
            else:
                summary['model_performance'][model_name]['fake_count'] += 1
    
    # Calculate averages
    if confidences:
        summary['avg_confidence'] = np.mean(confidences)
    
    # Calculate model averages
    for model in summary['model_performance']:
        if summary['model_performance'][model]['count'] > 0:
            summary['model_performance'][model]['avg_confidence'] /= summary['model_performance'][model]['count']
    
    return summary

def save_analysis_session(reports, session_name=None):
    """
    Save analysis session data to file
    """
    if session_name is None:
        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    session_data = {
        'session_name': session_name,
        'timestamp': datetime.now().isoformat(),
        'reports': reports,
        'summary': create_analysis_summary(reports)
    }
    
    filename = f"{session_name}.json"
    with open(filename, 'w') as f:
        json.dump(session_data, f, indent=2, default=str)
    
    return filename

def load_analysis_session(filename):
    """
    Load analysis session data from file
    """
    try:
        with open(filename, 'r') as f:
            session_data = json.load(f)
        return session_data
    except Exception as e:
        print(f"Error loading session: {e}")
        return None

def create_performance_metrics(reports):
    """
    Calculate detailed performance metrics
    """
    if not reports:
        return {}
    
    metrics = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1_score': 0,
        'processing_times': [],
        'confidence_correlation': 0
    }
    
    # Calculate processing times and confidence correlation
    processing_times = []
    confidences = []
    
    for report in reports:
        if 'timestamp' in report:
            # This would need to be enhanced with actual processing time tracking
            pass
        
        if 'prediction' in report and 'confidence' in report['prediction']:
            confidences.append(report['prediction']['confidence'])
    
    if confidences:
        metrics['avg_confidence'] = np.mean(confidences)
        metrics['confidence_std'] = np.std(confidences)
        metrics['min_confidence'] = min(confidences)
        metrics['max_confidence'] = max(confidences)
    
    return metrics 