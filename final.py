# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import logging
from ultralytics import YOLO
import cv2
from PIL import Image
import pytesseract
import re
import numpy as np
import sqlite3

DB_PATH = "pothole_history.db"

def init_db():
    """Initialize the database and create table if not exists"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS potholes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL,
            longitude REAL,
            count INTEGER DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()

init_db()  # Call this when app starts

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Configure upload folder
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.getcwd(), 'static', 'uploads'))
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load YOLO model
try:
    model = YOLO('C:/Users/Admin/Pictures/Final Capstone/final/best.pt')
except Exception as e:
    app.logger.error(f"Failed to load YOLO model: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_coordinates_from_text(text):
    """Extract latitude and longitude from OCR text using simple pattern matching"""
    try:
        # Specific pattern for "Lat X° Long Y°" format
        coord_pattern = r"Lat\s*([\d\.\-]+)[°]?\s*Long\s*([\d\.\-]+)[°]?"
        match = re.search(coord_pattern, text)

        if match:
            latitude = float(match.group(1))
            longitude = float(match.group(2))
            app.logger.debug(f"Found coordinates: Lat {latitude}, Long {longitude}")
            return {'latitude': latitude, 'longitude': longitude}
        
        app.logger.debug(f"No coordinates found in text: {text}")
        return None
        
    except Exception as e:
        app.logger.error(f"Error extracting coordinates: {str(e)}")
        return None

from PIL import Image
import exifread

def get_location_data(image_path):
    """Extract GPS coordinates from EXIF metadata or OCR."""
    try:
        # Open image and read EXIF data
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)

        lat_ref = tags.get('GPS GPSLatitudeRef')
        lat = tags.get('GPS GPSLatitude')
        lon_ref = tags.get('GPS GPSLongitude')
        lon = tags.get('GPS GPSLongitude')

        if lat and lon and lat_ref and lon_ref:
            def convert_to_degrees(value):
                d, m, s = [float(x.num) / float(x.den) for x in value.values]
                return d + (m / 60.0) + (s / 3600.0)

            latitude = convert_to_degrees(lat)
            if lat_ref.values[0] != 'N':
                latitude = -latitude

            longitude = convert_to_degrees(lon)
            if lon_ref.values[0] != 'E':
                longitude = -longitude

            return {'latitude': latitude, 'longitude': longitude, 'source': 'exif'}

        # If no EXIF GPS found, use OCR
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        ocr_coords = extract_coordinates_from_text(text)

        if ocr_coords:
            ocr_coords['source'] = 'ocr'
            return ocr_coords

    except Exception as e:
        app.logger.error(f"Error extracting GPS data: {str(e)}")

    return None  # If no GPS data is found

def detect_potholes(image_path):
    """Detect potholes using YOLO and extract GPS coordinates"""
    try:
        app.logger.debug(f"Processing image with YOLO: {image_path}")

        # Extract location data using OCR
        location_data = get_location_data(image_path)
        app.logger.debug(f"Location data extracted: {location_data}")

        # Run YOLO detection
        results = model(image_path)
        result = results[0]

        # Load the original image to draw detections
        image = cv2.imread(image_path)

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{confidence:.2f}', 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        (0, 255, 0), 
                        2)

        output_filename = 'detected_' + os.path.basename(image_path)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        if not cv2.imwrite(output_path, image):
            raise Exception(f"Failed to save result image to {output_path}")

        return len(result.boxes), output_filename, location_data

    except Exception as e:
        app.logger.error(f"Error in detect_potholes: {str(e)}")
        raise

def is_same_location(lat1, lon1, lat2, lon2, threshold=0.0001):
    """
    Check if two coordinates are within threshold distance
    threshold of 0.0001 is approximately 11 meters
    """
    return abs(lat1 - lat2) < threshold and abs(lon1 - lon2) < threshold

def update_pothole_database(latitude, longitude):
    """Update database with pothole detection"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check for existing nearby potholes
        cursor.execute("""
            SELECT id, latitude, longitude, count 
            FROM potholes 
            WHERE ABS(latitude - ?) < 0.0001 
            AND ABS(longitude - ?) < 0.0001
        """, (latitude, longitude))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update count for existing pothole
            cursor.execute("""
                UPDATE potholes 
                SET count = count + 1 
                WHERE id = ?
            """, (existing[0],))
        else:
            # Insert new pothole
            cursor.execute("""
                INSERT INTO potholes (latitude, longitude, count)
                VALUES (?, ?, 1)
            """, (latitude, longitude))
            
        conn.commit()
    except Exception as e:
        app.logger.error(f"Database error: {str(e)}")
    finally:
        conn.close()

@app.route('/history')
def get_history():
    """Endpoint to retrieve pothole history"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT latitude, longitude, count FROM potholes")
        potholes = cursor.fetchall()
        
        return jsonify([
            {
                'latitude': row[0],
                'longitude': row[1],
                'count': row[2]
            }
            for row in potholes
        ])
    except Exception as e:
        app.logger.error(f"Error retrieving history: {str(e)}")
        return jsonify({'error': str(e)})
    finally:
        conn.close()



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if not os.path.exists(filepath):
                return jsonify({'error': 'Failed to save uploaded file'})

            return jsonify({
                'message': 'File uploaded successfully',
                'filename': filename
            })

        return jsonify({'error': 'Invalid file type'})

    except Exception as e:
        app.logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': str(e)})

# Modify the predict route to include database updates
# app.py
# ... (keep existing imports) ...

@app.route('/update-location', methods=['POST'])
def update_location():
    """Endpoint to handle live location updates"""
    try:
        data = request.json
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        if latitude is None or longitude is None:
            return jsonify({'error': 'Missing coordinates'}), 400

        # Store the live location pothole detection
        update_pothole_database(latitude, longitude)
        
        return jsonify({
            'message': 'Location updated successfully',
            'latitude': latitude,
            'longitude': longitude
        })
    except Exception as e:
        app.logger.error(f"Error updating location: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Modify the predict route to better handle live location
@app.route('/predict', methods=['POST'])
def predict():
    """Runs pothole detection and stores results"""
    try:
        filename = request.form.get('filename')
        live_lat = request.form.get('latitude')  # Get live coordinates from form
        live_lon = request.form.get('longitude')
        
        if not filename:
            return jsonify({'error': 'No filename provided'})

        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(input_path):
            return jsonify({'error': f'Input file not found: {input_path}'})

        num_potholes, output_filename, location_data = detect_potholes(input_path)

        result_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        if not os.path.exists(result_path):
            return jsonify({'error': f'Result image not found: {result_path}'})

        # If we have GPS data from the image, use that
        if location_data and location_data.get('latitude') and location_data.get('longitude'):
            update_pothole_database(
                location_data['latitude'],
                location_data['longitude']
            )
            response_data = {
                'message': f'{num_potholes} potholes detected',
                'result_image': output_filename,
                'gps_data': location_data
            }
        # If we have live location data, use that
        elif live_lat and live_lon:
            try:
                lat = float(live_lat)
                lon = float(live_lon)
                update_pothole_database(lat, lon)
                response_data = {
                    'message': f'{num_potholes} potholes detected',
                    'result_image': output_filename,
                    'gps_data': {
                        'latitude': lat,
                        'longitude': lon,
                        'source': 'live_location'
                    }
                }
            except ValueError:
                return jsonify({'error': 'Invalid coordinates format'})
        else:
            response_data = {
                'message': f'{num_potholes} potholes detected',
                'result_image': output_filename,
                'gps_data': {'source': 'live_location'}
            }

        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Error in predict: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/static/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)