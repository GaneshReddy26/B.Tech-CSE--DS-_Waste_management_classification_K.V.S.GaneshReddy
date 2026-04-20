import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
from gtts import gTTS
import tempfile
import datetime
import csv
import time
import base64
from io import BytesIO
import threading
import queue
import uuid

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, Response

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for real-time detection
detection_queue = queue.Queue()
current_detections = []
last_detection_time = 0
DETECTION_INTERVAL = 2  # Process every 2 seconds

# Helper: text-to-speech 
def speak_text(text: str):
    """Convert text to speech and return a temp file path."""
    tts = gTTS(text=text, lang='en')
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    return tmp_file.name

# Helper: Save feedback to CSV
def save_feedback(predicted_class, actual_class, confidence, image_info, feedback_notes=""):
    """Save user feedback to a CSV file."""
    feedback_file = "feedback.csv"
    file_exists = os.path.isfile(feedback_file)
    
    with open(feedback_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'predicted_class', 'actual_class', 'confidence', 'image_info', 'feedback_notes'])
        
        writer.writerow([
            datetime.datetime.now().isoformat(),
            predicted_class,
            actual_class,
            confidence,
            image_info,
            feedback_notes
        ])

# Helper: Object detection and classification
class WasteDetector:
    def __init__(self, model, class_names, confidence_threshold=0.5):
        self.model = model
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
    
    def detect_objects(self, image):
        """Detect multiple waste objects in an image using sliding window approach"""
        img = np.array(image)
        original_h, original_w = img.shape[:2]
        
        # Create sliding windows
        window_size = 224
        stride = 112  # 50% overlap
        
        detections = []
        
        for y in range(0, original_h - window_size + 1, stride):
            for x in range(0, original_w - window_size + 1, stride):
                # Extract window
                window = img[y:y+window_size, x:x+window_size]
                
                # Predict
                window_pil = Image.fromarray(window)
                window_processed = self.preprocess_image(window_pil)
                preds = self.model.predict(window_processed, verbose=0)
                confidence = float(np.max(preds))
                class_idx = int(np.argmax(preds, axis=1)[0])
                
                if confidence > self.confidence_threshold:
                    detections.append({
                        'bbox': (x, y, x+window_size, y+window_size),
                        'class': self.class_names[class_idx],
                        'confidence': confidence,
                        'class_idx': class_idx
                    })
        
        # Apply non-maximum suppression
        detections = self.non_max_suppression(detections)
        return detections
    
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        img_resized = image.resize((224, 224))
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    
    def non_max_suppression(self, detections, iou_threshold=0.3):
        """Apply non-maximum suppression to remove overlapping detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            detections = [det for det in detections if self.iou(current['bbox'], det['bbox']) < iou_threshold]
        
        return keep
    
    def iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def get_color(self, class_idx):
        """Get distinct color for each class"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0)
        ]
        return colors[class_idx % len(colors)]

# Paths 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "final_waste_model_resnet.keras")

# Try multiple possible model locations
possible_model_paths = [
    model_path,
    os.path.join(BASE_DIR, "final_waste_model_resnet.keras"),
    os.path.join(BASE_DIR, "model", "final_waste_model_resnet.keras"),
    os.path.join(BASE_DIR, "..", "models", "final_waste_model_resnet.keras"),
    os.path.join(BASE_DIR, "..", "final_waste_model_resnet.keras"),
]

# Load model with fallback
def load_cached_model():
    model = None
    model_path_used = None
    
    for path in possible_model_paths:
        if os.path.exists(path):
            try:
                print(f"Loading model from: {path}")
                model = load_model(path)
                model_path_used = path
                print("Model loaded successfully!")
                break
            except Exception as e:
                print(f"Error loading model from {path}: {e}")
                continue
    
    if model is None:
        print("WARNING: No model file found. Using demo mode.")
        print("Available files in current directory:")
        for file in os.listdir(BASE_DIR):
            print(f"  - {file}")
        if os.path.exists(os.path.join(BASE_DIR, "models")):
            print("Files in models directory:")
            for file in os.listdir(os.path.join(BASE_DIR, "models")):
                print(f"  - {file}")
    
    return model, model_path_used

# Load model
model, model_path_used = load_cached_model()

# Define class names manually as fallback
class_names = [
    "cardboard waste", "clothe waste", "Electronic waste", "glass waste", 
    "metal waste", "organic waste", "paper waste", "plastic waste", 
    "shoes waste", "trash"
]

# Initialize waste detector (will work in demo mode if model is None)
waste_detector = WasteDetector(model, class_names, confidence_threshold=0.6) if model else None

# Recycling info mapping
recycling_info = {
    "Electronic waste": {
        "status": "Not recyclable in regular bins",
        "notes": (
            "E-waste (computers, phones, TVs, appliances) contains hazardous "
            "materials and valuable components. Do not throw it in household trash "
            "or curbside recycling. Take it to certified e-waste recycling centers, "
            "manufacturer take-back programs, or municipal collection events for "
            "safe handling."
        )
    },
    "cardboard waste": {
        "status": "Recyclable",
        "notes": (
            "Flatten boxes and keep them clean and dry before placing them in the "
            "recycling bin. Greasy or food-soiled cardboard should go into compost "
            "or trash."
        )
    },
    "clothe waste": {
        "status": "Recyclable/Reusable",
        "notes": (
            "Clean, wearable clothing can be donated to charities or thrift stores. "
            "Unwearable textiles can go to textile-recycling programs; most curbside "
            "programs don't accept them."
        )
    },
    "glass waste": {
        "status": "Recyclable",
        "notes": (
            "Rinse bottles and jars and remove lids before recycling. Broken window "
            "glass, mirrors, and ceramics are typically not accepted in glass "
            "recycling streams."
        )
    },
    "metal waste": {
        "status": "Recyclable",
        "notes": (
            "Empty and rinse aluminum and steel cans before recycling. Large scrap "
            "metal pieces should be taken to a scrap yard or municipal drop-off "
            "site, not put in curbside bins."
        )
    },
    "organic waste": {
        "status": "Compostable (not in recycling)",
        "notes": (
            "Food scraps, yard trimmings, and other biodegradable items should go "
            "into a compost bin or green-waste collection, not into the recycling bin."
        )
    },
    "paper waste": {
        "status": "Recyclable",
        "notes": (
            "Recycle clean and dry newspapers, office paper, and magazines. Shredded "
            "paper, wax-coated, or heavily soiled paper may require special handling "
            "or belong in the trash/compost."
        )
    },
    "plastic waste": {
        "status": "Some types recyclable",
        "notes": (
            "Check the resin code on the item (numbers 1–7). Most curbside programs "
            "accept bottles and jugs (#1 and #2). Plastic bags, films, and Styrofoam "
            "usually need special drop-off locations."
        )
    },
    "shoes waste": {
        "status": "Recyclable via special programs",
        "notes": (
            "Donate wearable shoes to charities or take any shoes to specialized shoe "
            "recycling programs (often run by brands or stores). Curbside recycling "
            "usually does not accept shoes."
        )
    },
    "trash": {
        "status": "Not recyclable",
        "notes": (
            "Mixed, contaminated, or non-recyclable waste must be placed in regular "
            "landfill bins. Reduce, reuse, or separate recyclable components whenever "
            "possible before disposal."
        )
    }
}

class RealTimeDetector:
    def __init__(self, waste_detector):
        self.waste_detector = waste_detector
        self.is_running = False
        self.current_frame = None
        self.lock = threading.Lock()
    
    def start(self):
        self.is_running = True
    
    def stop(self):
        self.is_running = False
    
    def process_frame(self, frame):
        """Process a single frame for waste detection"""
        if not self.is_running:
            return frame, []
        
        try:
            # Convert frame to PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Detect objects
            detections = self.waste_detector.detect_objects(pil_image)
            
            # Draw bounding boxes and labels
            annotated_frame = frame.copy()
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                class_name = detection['class']
                confidence = detection['confidence']
                
                # Draw bounding box
                color = self.waste_detector.get_color(detection['class_idx'])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return annotated_frame, detections
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame, []

def generate_frames():
    """Generate video frames with real-time detection"""
    camera = cv2.VideoCapture(0)
    detector = RealTimeDetector(waste_detector)
    detector.start()
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            # Process frame for detection
            processed_frame, detections = detector.process_frame(frame)
            
            # Update global detections
            global current_detections, last_detection_time
            current_time = time.time()
            if current_time - last_detection_time > DETECTION_INTERVAL and detections:
                current_detections = detections
                last_detection_time = current_time
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        print(f"Error in video stream: {e}")
    finally:
        detector.stop()
        camera.release()

def generate_demo_frames():
    """Generate demo video frames when model is not available"""
    camera = cv2.VideoCapture(0)
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            # Add demo bounding boxes and text
            demo_frame = frame.copy()
            height, width = demo_frame.shape[:2]
            
            # Draw demo bounding box
            box_color = (0, 255, 0)
            box_thickness = 3
            x1, y1 = width // 4, height // 4
            x2, y2 = 3 * width // 4, 3 * height // 4
            cv2.rectangle(demo_frame, (x1, y1), (x2, y2), box_color, box_thickness)
            
            # Draw demo label
            label = "Demo: plastic waste - 0.85"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(demo_frame, (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), box_color, -1)
            cv2.putText(demo_frame, label, (x1, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add demo mode text
            demo_text = "DEMO MODE - Point camera at waste objects"
            text_size = cv2.getTextSize(demo_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = (width - text_size[0]) // 2
            cv2.putText(demo_frame, demo_text, (text_x, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Update global detections with demo data
            global current_detections, last_detection_time
            current_time = time.time()
            if current_time - last_detection_time > DETECTION_INTERVAL:
                current_detections = [{
                    'class': 'plastic waste',
                    'confidence': 0.85,
                    'status': 'Some types recyclable',
                    'notes': recycling_info.get('plastic waste', {}).get('notes', 'Demo mode')
                }]
                last_detection_time = current_time
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', demo_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        print(f"Error in demo video stream: {e}")
    finally:
        camera.release()

def init_session():
    """Initialize session variables"""
    if 'prediction_history' not in session:
        session['prediction_history'] = []
    if 'uploaded_image' not in session:
        session['uploaded_image'] = None
    if 'detection_results' not in session:
        session['detection_results'] = None

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def plot_to_base64():
    """Convert matplotlib plot to base64 string"""
    buffered = BytesIO()
    plt.savefig(buffered, format='png', bbox_inches='tight')
    buffered.seek(0)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{img_str}"

def demo_prediction():
    """Generate demo predictions when model is not available"""
    # Return a random prediction for demo purposes
    import random
    demo_class = random.choice(class_names)
    confidence = random.uniform(0.7, 0.95)
    
    return [{
        'object': 1,
        'class': demo_class,
        'confidence': f"{confidence*100:.1f}%",
        'status': recycling_info.get(demo_class, {}).get('status', 'Unknown'),
        'notes': recycling_info.get(demo_class, {}).get('notes', 'No information available')
    }]

@app.route('/')
def index():
    init_session()
    return render_template('index.html', model_loaded=model is not None)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    init_session()
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error='No file selected', model_loaded=model is not None)
        
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error='No file selected', model_loaded=model is not None)
        
        if file:
            try:
                # Save uploaded file
                filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process image
                img = Image.open(filepath).convert('RGB')
                session['uploaded_image'] = image_to_base64(img)
                
                if model is None:
                    # Demo mode - generate fake detections
                    detection_data = demo_prediction()
                    session['detection_results'] = detection_data
                    
                    # Add to history
                    for detection in detection_data:
                        history_entry = {
                            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'image': f"Demo_Detection",
                            'predicted_class': detection['class'],
                            'confidence': detection['confidence'],
                            'status': detection['status']
                        }
                        session['prediction_history'].append(history_entry)
                        session.modified = True
                    
                    return render_template('upload.html', 
                                        uploaded_image=session['uploaded_image'],
                                        detections=detection_data,
                                        demo_mode=True,
                                        model_loaded=False)
                
                # Multi-object detection with actual model
                detections = waste_detector.detect_objects(img)
                
                if detections:
                    # Create annotated image
                    img_annotated = np.array(img.copy())
                    for detection in detections:
                        x1, y1, x2, y2 = detection['bbox']
                        class_name = detection['class']
                        confidence = detection['confidence']
                        
                        # Draw bounding box
                        color = waste_detector.get_color(detection['class_idx'])
                        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 3)
                        
                        # Draw label
                        label = f"{class_name}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(img_annotated, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(img_annotated, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    annotated_img = Image.fromarray(img_annotated)
                    session['annotated_image'] = image_to_base64(annotated_img)
                    
                    # Prepare detection results
                    detection_data = []
                    for i, detection in enumerate(detections):
                        class_name = detection['class']
                        info = recycling_info.get(class_name, {})
                        detection_data.append({
                            'object': i+1,
                            'class': class_name,
                            'confidence': f"{detection['confidence']*100:.1f}%",
                            'status': info.get('status', 'Unknown'),
                            'notes': info.get('notes', 'No information available')
                        })
                    
                    session['detection_results'] = detection_data
                    
                    # Add to history
                    for detection in detections:
                        history_entry = {
                            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'image': f"Multi_Object_Detection",
                            'predicted_class': detection['class'],
                            'confidence': f"{detection['confidence']*100:.1f}%",
                            'status': recycling_info.get(detection['class'], {}).get('status', 'Unknown')
                        }
                        session['prediction_history'].append(history_entry)
                        session.modified = True
                    
                    return render_template('upload.html', 
                                        uploaded_image=session['uploaded_image'],
                                        annotated_image=session['annotated_image'],
                                        detections=detection_data,
                                        model_loaded=True)
                
                else:
                    # Fallback to single prediction
                    img_resized = img.resize((224, 224))
                    img_array = img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)

                    # Predict
                    preds = model.predict(img_array, verbose=0)
                    class_idx = int(np.argmax(preds, axis=1)[0])
                    confidence = float(np.max(preds))
                    predicted_class = class_names[class_idx]

                    # Get top 3 predictions
                    top3_idx = np.argsort(preds[0])[::-1][:3]
                    top3_predictions = [(class_names[i], float(preds[0][i])) for i in top3_idx]

                    # Create confidence plot
                    plt.figure(figsize=(10, 6))
                    y_pos = np.arange(len(class_names))
                    plt.barh(y_pos, preds[0] * 100)
                    plt.yticks(y_pos, class_names)
                    plt.xlabel('Confidence (%)')
                    plt.title('Prediction Confidence for All Classes')
                    plt.tight_layout()
                    confidence_plot = plot_to_base64()

                    info = recycling_info.get(predicted_class, {})
                    detection_data = [{
                        'object': 1,
                        'class': predicted_class,
                        'confidence': f"{confidence*100:.1f}%",
                        'status': info.get('status', 'Unknown'),
                        'notes': info.get('notes', 'No information available')
                    }]

                    session['detection_results'] = detection_data
                    session['confidence_plot'] = confidence_plot
                    session['top3_predictions'] = top3_predictions

                    # Add to history
                    history_entry = {
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'image': f"Single_Image_{len(session['prediction_history']) + 1}",
                        'predicted_class': predicted_class,
                        'confidence': f"{confidence*100:.1f}%",
                        'status': info.get('status', 'Unknown')
                    }
                    session['prediction_history'].append(history_entry)
                    session.modified = True

                    return render_template('upload.html',
                                        uploaded_image=session['uploaded_image'],
                                        confidence_plot=confidence_plot,
                                        top3_predictions=top3_predictions,
                                        detections=detection_data,
                                        model_loaded=True)
            
            except Exception as e:
                return render_template('upload.html', error=f'Error processing image: {str(e)}', model_loaded=model is not None)
    
    return render_template('upload.html', model_loaded=model is not None)

@app.route('/camera', methods=['GET', 'POST'])
def camera():
    init_session()
    
    if request.method == 'POST':
        # Handle captured image from camera
        image_data = request.form.get('image_data')
        if image_data:
            try:
                # Remove data URL prefix
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                
                # Save image
                filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_capture.jpg")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
                
                # Store in session and redirect to upload for processing
                session['camera_image_path'] = filepath
                return redirect(url_for('process_camera_image'))
            
            except Exception as e:
                return render_template('camera.html', error=f'Error capturing image: {str(e)}', model_loaded=model is not None)
    
    return render_template('camera.html', model_loaded=model is not None)

@app.route('/process_camera_image')
def process_camera_image():
    """Process the captured camera image"""
    if 'camera_image_path' not in session or not os.path.exists(session['camera_image_path']):
        return redirect(url_for('camera'))
    
    try:
        # Load and process the captured image
        img = Image.open(session['camera_image_path']).convert('RGB')
        session['uploaded_image'] = image_to_base64(img)
        
        if model is None:
            # Demo mode
            detection_data = demo_prediction()
        else:
            # Actual model prediction
            detections = waste_detector.detect_objects(img)
            if detections:
                detection_data = []
                for i, detection in enumerate(detections):
                    class_name = detection['class']
                    info = recycling_info.get(class_name, {})
                    detection_data.append({
                        'object': i+1,
                        'class': class_name,
                        'confidence': f"{detection['confidence']*100:.1f}%",
                        'status': info.get('status', 'Unknown'),
                        'notes': info.get('notes', 'No information available')
                    })
            else:
                # Single prediction fallback
                img_resized = img.resize((224, 224))
                img_array = img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                preds = model.predict(img_array, verbose=0)
                class_idx = int(np.argmax(preds, axis=1)[0])
                confidence = float(np.max(preds))
                predicted_class = class_names[class_idx]
                
                info = recycling_info.get(predicted_class, {})
                detection_data = [{
                    'object': 1,
                    'class': predicted_class,
                    'confidence': f"{confidence*100:.1f}%",
                    'status': info.get('status', 'Unknown'),
                    'notes': info.get('notes', 'No information available')
                }]
        
        session['detection_results'] = detection_data
        
        # Add to history
        for detection in detection_data:
            history_entry = {
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'image': f"Camera_Capture",
                'predicted_class': detection['class'],
                'confidence': detection['confidence'],
                'status': detection['status']
            }
            session['prediction_history'].append(history_entry)
        
        session.modified = True
        return render_template('upload.html',
                            uploaded_image=session['uploaded_image'],
                            detections=detection_data,
                            model_loaded=model is not None,
                            demo_mode=model is None)
    
    except Exception as e:
        return render_template('camera.html', error=f'Error processing image: {str(e)}', model_loaded=model is not None)


@app.route('/realtime')
def realtime():
    init_session()
    return render_template('realtime.html', model_loaded=model is not None)
# Add this route for video feed
@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    if model is not None:
        return Response(generate_frames(),
                      mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(generate_demo_frames(),
                      mimetype='multipart/x-mixed-replace; boundary=frame')

# Add this route for getting current detections
@app.route('/get_detections')
def get_detections():
    """Get current detection results"""
    global current_detections
    return jsonify(current_detections)

# Add this route for capturing frame
@app.route('/capture_frame')
def capture_frame():
    """Capture current frame and save it"""
    try:
        camera = cv2.VideoCapture(0)
        success, frame = camera.read()
        camera.release()
        
        if success:
            # Save captured frame
            filename = f"capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv2.imwrite(filepath, frame)
            
            # Process the captured image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            if model is not None:
                detections = waste_detector.detect_objects(pil_image)
                if detections:
                    detection_data = []
                    for i, detection in enumerate(detections):
                        class_name = detection['class']
                        info = recycling_info.get(class_name, {})
                        detection_data.append({
                            'object': i+1,
                            'class': class_name,
                            'confidence': f"{detection['confidence']*100:.1f}%",
                            'status': info.get('status', 'Unknown'),
                            'notes': info.get('notes', 'No information available')
                        })
                else:
                    # Single prediction fallback
                    img_resized = pil_image.resize((224, 224))
                    img_array = img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)
                    preds = model.predict(img_array, verbose=0)
                    class_idx = int(np.argmax(preds, axis=1)[0])
                    confidence = float(np.max(preds))
                    predicted_class = class_names[class_idx]
                    
                    info = recycling_info.get(predicted_class, {})
                    detection_data = [{
                        'object': 1,
                        'class': predicted_class,
                        'confidence': f"{confidence*100:.1f}%",
                        'status': info.get('status', 'Unknown'),
                        'notes': info.get('notes', 'No information available')
                    }]
            else:
                # Demo mode
                detection_data = demo_prediction()
            
            return jsonify({
                'success': True,
                'image_url': f"/uploads/{filename}",
                'detections': detection_data
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to capture frame'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/history')
def history():
    init_session()
    return render_template('history.html', history=session['prediction_history'], model_loaded=model is not None)

@app.route('/clear_history')
def clear_history():
    session['prediction_history'] = []
    session.modified = True
    return redirect(url_for('history'))

@app.route('/export_history')
def export_history():
    if not session.get('prediction_history'):
        return redirect(url_for('history'))
    
    df = pd.DataFrame(session['prediction_history'])
    csv_data = df.to_csv(index=False)
    
    # Create response with CSV file
    output = BytesIO()
    output.write(csv_data.encode('utf-8'))
    output.seek(0)
    
    filename = f"waste_classification_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    
    return send_file(output, 
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name=filename)

@app.route('/get_speech/<class_name>')
def get_speech(class_name):
    """Generate speech for a specific waste class"""
    info = recycling_info.get(class_name, {})
    if info:
        speech_text = f"{class_name}. {info['status']}. {info['notes']}"
        speech_file = speak_text(speech_text)
        return send_file(speech_file, as_attachment=True, download_name=f"{class_name}_info.mp3")
    return "No information available", 404

@app.route('/model_status')
def model_status():
    """Check model status"""
    return jsonify({
        'model_loaded': model is not None,
        'model_path': model_path_used if model else None,
        'class_names': class_names
    })

def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    print("=" * 50)
    print("Waste Classification Flask App")
    print("=" * 50)
    if model:
        print(f"✅ Model loaded successfully from: {model_path_used}")
    else:
        print("⚠️  Running in DEMO MODE - No model file found")
        print("Please place your model file in one of these locations:")
        for path in possible_model_paths:
            print(f"  - {path}")
    print("=" * 50)
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)