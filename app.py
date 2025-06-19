import os
import tempfile
import uuid
import cv2
from flask import Flask, jsonify, render_template, send_file, redirect, request
from werkzeug.utils import secure_filename
from OBR import SegmentationEngine, BrailleClassifier, BrailleImage
import pyttsx3
from pydub import AudioSegment

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
tempdir = tempfile.TemporaryDirectory()
app = Flask("Optical Braille Recognition Demo")
app.config['UPLOAD_FOLDER'] = tempdir.name

AUDIO_PATH = os.path.join(app.config['UPLOAD_FOLDER'], "tts.mp3")

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
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        classifier = BrailleClassifier()
        img = BrailleImage(image_path)
        for letter in SegmentationEngine(image=img):
            letter.mark()
            classifier.push(letter)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}-proc.png"), img.get_final_image())
        os.unlink(image_path)

        digest_result = classifier.digest()
        return jsonify({
            "error": False,
            "message": "Processed and Digested successfully",
            "img_id": filename,
            "digest": digest_result
        })

@app.route('/speech', methods=['POST'])
def speech():
    text = request.form.get("text", "")
    if not text:
        return jsonify({"error": True, "message": "No text provided"})

    wav_path = os.path.join(app.config['UPLOAD_FOLDER'], "tts.wav")

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.save_to_file(text, wav_path)
    engine.runAndWait()

    sound = AudioSegment.from_wav(wav_path)
    sound.export(AUDIO_PATH, format="mp3")
    os.remove(wav_path)

    return jsonify({"error": False, "url": "/getaudio"})

@app.route('/getaudio')
def get_audio():
    if os.path.exists(AUDIO_PATH):
        return send_file(AUDIO_PATH, mimetype="audio/mpeg")
    return jsonify({"error": True, "message": "Audio not found"})

if __name__ == "__main__":
    app.run()
