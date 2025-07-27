from flask import Flask, request, jsonify
from models.yolo import detect_objects
from models.captioning import generate_caption
from PIL import Image
import io

app = Flask(__name__)

@app.route('/caption', methods=['POST'])
def caption_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")

    # Step 1: Object detection (optional to include in caption)
    detected_objects = detect_objects(image)

    # Step 2: Generate caption using BLIP
    caption = generate_caption(image)

    return jsonify({
        "caption": caption,
        "objects": detected_objects
    })

if __name__ == '__main__':
    app.run(debug=True)
