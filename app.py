import os
import tempfile
import uuid
import cv2
from flask import Flask, jsonify, render_template, send_file, redirect, request
from werkzeug.utils import secure_filename
from OBR import SegmentationEngine, BrailleClassifier, BrailleImage
import edge_tts
import asyncio

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
tempdir = tempfile.TemporaryDirectory()
app = Flask("Optical Braille Recognition Demo")
app.config['UPLOAD_FOLDER'] = tempdir.name

# Dynamic audio filename will be used per request
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/favicon.ico')
def fav():
    return send_file('favicon.ico', mimetype='image/ico')

@app.route('/coverimage')
def cover_image():
    return send_file('samples/sample1.png', mimetype='image/png')

@app.route('/procimage/<string:img_id>')
def proc_image(img_id):
    image = '{}/{}-proc.png'.format(tempdir.name, secure_filename(img_id))
    if os.path.exists(image) and os.path.isfile(image):
        return send_file(image, mimetype='image/png')
    return redirect('/coverimage')

@app.route('/digest', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": True, "message": "file not in request"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": True, "message": "empty filename"})
    if file and allowed_file(file.filename):
        filename = ''.join(str(uuid.uuid4()).split('-'))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        classifier = BrailleClassifier()
        img = BrailleImage(file_path)
        for letter in SegmentationEngine(image=img):
            letter.mark()
            classifier.push(letter)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}-proc.png"), img.get_final_image())
        os.unlink(file_path)

        digest_result = classifier.digest()
        return jsonify({
            "error": False,
            "message": "Processed and Digested successfully",
            "img_id": filename,
            "digest": digest_result
        })

@app.route('/speech', methods=['POST'])
def speech():
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": True, "message": "No text provided"})

    audio_filename = f"{uuid.uuid4()}.mp3"
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)

    async def generate_tts():
        communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
        await communicate.save(audio_path)

    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            task = loop.create_task(generate_tts())
            loop.run_until_complete(task)
        else:
            loop.run_until_complete(generate_tts())

        return jsonify({"error": False, "url": f"/getaudio/{audio_filename}"})
    except Exception as e:
        return jsonify({"error": True, "message": f"TTS failed: {str(e)}"})

@app.route('/getaudio/<string:filename>')
def get_audio(filename):
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype="audio/mpeg")
    return jsonify({"error": True, "message": "Audio not found"})

if __name__ == "__main__":
    app.run(debug=True)
