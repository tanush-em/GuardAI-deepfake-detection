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
import face_recognition
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

def detect_fake_regions(frame: np.ndarray, face_landmarks: List, fake_probability: float, 
                       confidence_threshold: float = 0.7) -> List[Dict]:
    """
    Detect potential fake regions in a frame based on face landmarks and fake probability
    
    Args:
        frame: Input frame as numpy array
        face_landmarks: Face landmarks from face_recognition
        fake_probability: Probability that the frame is fake
        confidence_threshold: Threshold for considering regions as fake
    
    Returns:
        List of dictionaries containing fake region information
    """
    fake_regions = []
    
    # Only proceed if fake probability is above threshold
    if fake_probability < confidence_threshold:
        return fake_regions
    
    if not face_landmarks:
        return fake_regions
    
    # Get the first face landmarks
    landmarks = face_landmarks[0]
    
    # Define regions of interest for deepfake detection using face_recognition landmark keys
    regions_of_interest = {
        'eyes': {
            'landmark_keys': ['left_eye', 'right_eye'],
            'name': 'Eyes Region',
            'suspicion_factor': 0.8
        },
        'nose': {
            'landmark_keys': ['nose_bridge', 'nose_tip'],
            'name': 'Nose Region',
            'suspicion_factor': 0.6
        },
        'mouth': {
            'landmark_keys': ['top_lip', 'bottom_lip'],
            'name': 'Mouth Region',
            'suspicion_factor': 0.7
        },
        'face_boundary': {
            'landmark_keys': ['chin'],
            'name': 'Face Boundary',
            'suspicion_factor': 0.9
        },
        'cheeks': {
            'landmark_keys': ['left_eyebrow', 'right_eyebrow'],
            'name': 'Cheek Region',
            'suspicion_factor': 0.5
        }
    }
    
    for region_name, region_info in regions_of_interest.items():
        # Get landmarks for this region
        region_landmarks = []
        for key in region_info['landmark_keys']:
            if key in landmarks:
                region_landmarks.extend(landmarks[key])
        
        if len(region_landmarks) < 3:  # Need at least 3 points for a meaningful region
            continue
        
        # Calculate bounding box for the region
        x_coords = [point[0] for point in region_landmarks]
        y_coords = [point[1] for point in region_landmarks]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding to the bounding box
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)
        
        # Calculate region suspicion score based on fake probability and region-specific factors
        suspicion_score = fake_probability * region_info['suspicion_factor']
        
        # Only include regions with high suspicion
        if suspicion_score > confidence_threshold * 0.8:
            fake_regions.append({
                'region_name': region_info['name'],
                'bbox': (x_min, y_min, x_max, y_max),
                'landmarks': region_landmarks,
                'suspicion_score': suspicion_score,
                'fake_probability': fake_probability
            })
    
    return fake_regions

def draw_fake_region_annotations(frame: np.ndarray, fake_regions: List[Dict], 
                                show_landmarks: bool = True, show_confidence: bool = True) -> np.ndarray:
    """
    Draw bounding boxes and annotations for fake regions
    
    Args:
        frame: Input frame as numpy array
        fake_regions: List of fake region dictionaries
        show_landmarks: Whether to show individual landmarks
        show_confidence: Whether to show confidence scores
    
    Returns:
        Annotated frame as numpy array
    """
    annotated_frame = frame.copy()
    
    # Color scheme for different regions
    region_colors = {
        'Eyes Region': (255, 0, 0),      # Red
        'Nose Region': (0, 255, 0),      # Green
        'Mouth Region': (0, 0, 255),     # Blue
        'Face Boundary': (255, 255, 0),  # Yellow
        'Cheek Region': (255, 0, 255)    # Magenta
    }
    
    for region in fake_regions:
        bbox = region['bbox']
        region_name = region['region_name']
        suspicion_score = region['suspicion_score']
        
        # Get color for this region
        color = region_colors.get(region_name, (255, 255, 255))
        
        # Draw bounding box
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw region name and confidence
        if show_confidence:
            label = f"{region_name}: {suspicion_score:.2f}"
        else:
            label = region_name
        
        # Calculate text position
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x_min
        text_y = y_min - 10 if y_min > 20 else y_max + 20
        
        # Draw text background
        cv2.rectangle(annotated_frame, 
                     (text_x, text_y - text_size[1] - 5),
                     (text_x + text_size[0], text_y + 5),
                     color, -1)
        
        # Draw text
        cv2.putText(annotated_frame, label, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw landmarks if requested
        if show_landmarks:
            for landmark in region['landmarks']:
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(annotated_frame, (x, y), 2, color, -1)
    
    return annotated_frame

def create_fake_region_heatmap(frame: np.ndarray, fake_regions: List[Dict], 
                              heatmap_resolution: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Create a heatmap showing the concentration of fake regions
    
    Args:
        frame: Input frame as numpy array
        fake_regions: List of fake region dictionaries
        heatmap_resolution: Resolution of the output heatmap
    
    Returns:
        Heatmap as numpy array
    """
    # Create empty heatmap
    heatmap = np.zeros(heatmap_resolution, dtype=np.float32)
    
    if not fake_regions:
        return heatmap
    
    # Scale factors
    scale_x = heatmap_resolution[1] / frame.shape[1]
    scale_y = heatmap_resolution[0] / frame.shape[0]
    
    for region in fake_regions:
        bbox = region['bbox']
        suspicion_score = region['suspicion_score']
        
        # Scale bounding box to heatmap resolution
        x_min = int(bbox[0] * scale_x)
        y_min = int(bbox[1] * scale_y)
        x_max = int(bbox[2] * scale_x)
        y_max = int(bbox[3] * scale_y)
        
        # Add suspicion score to heatmap region
        heatmap[y_min:y_max, x_min:x_max] += suspicion_score
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap

def analyze_frame_artifacts(frame: np.ndarray, face_landmarks: List) -> Dict:
    """
    Analyze frame for common deepfake artifacts
    
    Args:
        frame: Input frame as numpy array
        face_landmarks: Face landmarks from face_recognition
    
    Returns:
        Dictionary containing artifact analysis results
    """
    artifacts = {
        'compression_artifacts': 0.0,
        'lighting_inconsistencies': 0.0,
        'edge_artifacts': 0.0,
        'texture_anomalies': 0.0,
        'overall_artifact_score': 0.0
    }
    
    if not face_landmarks:
        return artifacts
    
    # Get face region
    landmarks = face_landmarks[0]
    # Flatten all landmark points
    all_points = []
    for key, points in landmarks.items():
        all_points.extend(points)
    
    x_coords = [point[0] for point in all_points]
    y_coords = [point[1] for point in all_points]
    
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    
    # Add padding
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(frame.shape[1], x_max + padding)
    y_max = min(frame.shape[0], y_max + padding)
    
    face_region = frame[y_min:y_max, x_min:x_max]
    
    if face_region.size == 0:
        return artifacts
    
    # Analyze compression artifacts (blocking, ringing)
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    
    # Ensure even dimensions for DCT (OpenCV requirement)
    h, w = gray_face.shape
    if h % 2 == 1:
        gray_face = gray_face[:-1, :]
    if w % 2 == 1:
        gray_face = gray_face[:, :-1]
    
    # DCT analysis for compression artifacts
    if gray_face.size > 0:
        dct = cv2.dct(np.float32(gray_face))
        dct_energy = np.sum(np.abs(dct[8:, 8:])) / np.sum(np.abs(dct))
        artifacts['compression_artifacts'] = min(1.0, dct_energy * 10)
    else:
        artifacts['compression_artifacts'] = 0.0
    
    # Analyze lighting inconsistencies
    lab_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
    l_channel = lab_face[:, :, 0]
    lighting_variance = np.var(l_channel)
    artifacts['lighting_inconsistencies'] = min(1.0, lighting_variance / 1000)
    
    # Analyze edge artifacts
    edges = cv2.Canny(gray_face, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    artifacts['edge_artifacts'] = min(1.0, edge_density * 5)
    
    # Analyze texture anomalies using local binary patterns
    # Simplified texture analysis
    texture_variance = np.var(cv2.Laplacian(gray_face, cv2.CV_64F))
    artifacts['texture_anomalies'] = min(1.0, texture_variance / 1000)
    
    # Calculate overall artifact score
    artifact_scores = [
        artifacts['compression_artifacts'],
        artifacts['lighting_inconsistencies'],
        artifacts['edge_artifacts'],
        artifacts['texture_anomalies']
    ]
    artifacts['overall_artifact_score'] = np.mean(artifact_scores)
    
    return artifacts

def create_fake_region_report(fake_regions: List[Dict], artifact_analysis: Dict, 
                            frame_metadata: Dict) -> Dict:
    """
    Create a comprehensive report for fake region analysis
    
    Args:
        fake_regions: List of fake region dictionaries
        artifact_analysis: Artifact analysis results
        frame_metadata: Frame metadata
    
    Returns:
        Dictionary containing the fake region report
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'frame_info': frame_metadata,
        'fake_regions_detected': len(fake_regions),
        'regions': [],
        'artifact_analysis': artifact_analysis,
        'overall_risk_score': 0.0,
        'recommendations': []
    }
    
    # Process each fake region
    for region in fake_regions:
        region_report = {
            'name': region['region_name'],
            'bbox': region['bbox'],
            'suspicion_score': region['suspicion_score'],
            'fake_probability': region['fake_probability'],
            'risk_level': 'High' if region['suspicion_score'] > 0.8 else 'Medium' if region['suspicion_score'] > 0.6 else 'Low'
        }
        report['regions'].append(region_report)
    
    # Calculate overall risk score
    if fake_regions:
        max_suspicion = max(region['suspicion_score'] for region in fake_regions)
        avg_suspicion = np.mean([region['suspicion_score'] for region in fake_regions])
        artifact_score = artifact_analysis['overall_artifact_score']
        
        report['overall_risk_score'] = (max_suspicion * 0.5 + avg_suspicion * 0.3 + artifact_score * 0.2)
    
    # Generate recommendations
    if report['overall_risk_score'] > 0.8:
        report['recommendations'].append("High risk of deepfake manipulation detected")
        report['recommendations'].append("Consider additional verification methods")
        report['recommendations'].append("Review video source and context")
    elif report['overall_risk_score'] > 0.6:
        report['recommendations'].append("Moderate risk of manipulation detected")
        report['recommendations'].append("Further analysis recommended")
    elif report['overall_risk_score'] > 0.4:
        report['recommendations'].append("Low risk of manipulation")
        report['recommendations'].append("Video appears authentic")
    else:
        report['recommendations'].append("No significant manipulation detected")
        report['recommendations'].append("Video appears to be authentic")
    
    return report

def create_fake_region_visualization(fake_regions: List[Dict], artifact_analysis: Dict) -> Dict:
    """
    Create visualizations for fake region analysis
    
    Args:
        fake_regions: List of fake region dictionaries
        artifact_analysis: Artifact analysis results
    
    Returns:
        Dictionary containing visualization plots
    """
    plots = {}
    
    # Create region suspicion bar chart
    if fake_regions:
        region_names = [region['region_name'] for region in fake_regions]
        suspicion_scores = [region['suspicion_score'] for region in fake_regions]
        
        fig_regions = go.Figure(data=[
            go.Bar(x=region_names, y=suspicion_scores,
                   marker_color=['red' if score > 0.8 else 'orange' if score > 0.6 else 'yellow' 
                               for score in suspicion_scores],
                   text=[f'{score:.3f}' for score in suspicion_scores],
                   textposition='auto')
        ])
        fig_regions.update_layout(
            title='Fake Region Suspicion Scores',
            xaxis_title='Face Regions',
            yaxis_title='Suspicion Score',
            height=400
        )
        plots['region_suspicion'] = fig_regions
    
    # Create artifact analysis radar chart
    artifact_names = list(artifact_analysis.keys())
    artifact_values = list(artifact_analysis.values())
    
    fig_artifacts = go.Figure(data=go.Scatterpolar(
        r=artifact_values,
        theta=artifact_names,
        fill='toself',
        name='Artifact Scores'
    ))
    fig_artifacts.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title='Deepfake Artifact Analysis',
        height=400
    )
    plots['artifact_radar'] = fig_artifacts
    
    return plots 