from flask import Flask, request, render_template, jsonify
import os
from PIL import Image
import io
from facade_calc_part_b import get_building_info

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def handle_image_upload():
    image = request.files['file']
    lat = request.form['lat']
    lon = request.form['lon']
    image_path = os.path.join('uploads', image.filename)
    image.save(image_path)

    building_info = get_building_info(float(lat), float(lon))

    # Ensure no NaN values in the JSON response
    if building_info['building_name'] != building_info['building_name']:  # This checks for NaN
        building_info['building_name'] = None

    response = {
        "segmented_image_path": "/static/Assets/01_segmentation.png",
        "classified_image_path": "/static/Assets/02_Classifier.png",
        "building_info": building_info,
        "message": "Image processed successfully"
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
