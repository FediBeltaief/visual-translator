from flask import Flask, render_template, request, jsonify
import os
import uuid
import easyocr
from transformers import MarianMTModel, MarianTokenizer
import cv2
import numpy as np
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

translation_models = {
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "ar": "Helsinki-NLP/opus-mt-en-ar",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "it": "Helsinki-NLP/opus-mt-en-it"
}

loaded_models = {}

def get_model(lang):
    if lang not in loaded_models:
        model_name = translation_models[lang]
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        loaded_models[lang] = (tokenizer, model)
    return loaded_models[lang]

def translate_text(text, lang):
    if not text or not text.strip():
        return ""
    try:
        tokenizer, model = get_model(lang)
        # Split by newlines to preserve paragraph structure
        lines = text.split('\n')
        translated_lines = []
        
        for line in lines:
            if not line.strip():
                translated_lines.append("")
                continue
            
            # Translate line by line
            tokens = tokenizer(line, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translation = model.generate(**tokens)
            decoded = tokenizer.decode(translation[0], skip_special_tokens=True)
            translated_lines.append(decoded)
            
        return "\n".join(translated_lines)
    except Exception as e:
        print(f"Translation error: {e}")
        return "Translation failed."

def process_image_and_extract(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, [], None

    scale = 2.0
    height, width = img.shape[:2]
    img_resized = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR)
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    results = ocr_reader.readtext(gray, paragraph=True)

    if not results:
        gray_inverted = cv2.bitwise_not(gray)
        results = ocr_reader.readtext(gray_inverted, paragraph=True)

    return img_resized, results

@app.route("/", methods=["GET", "POST"])
def index():
    translated = None
    extracted = None
    image_path = None
    target_lang = "fr" # default

    if request.method == "POST":
        file = request.files.get('image')
        target_lang = request.form.get("target_lang", "fr")

        if file and file.filename != '':
            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            processed_img, ocr_results = process_image_and_extract(filepath)

            if processed_img is not None:
                extracted_list = [item[1] for item in ocr_results]
                extracted = "\n\n".join(extracted_list)

                for (bbox, text) in ocr_results:
                    pts = np.array([bbox], dtype=np.int32)
                    cv2.polylines(processed_img, pts, isClosed=True, color=(0, 255, 0), thickness=3)

                boxed_filename = f"{uuid.uuid4()}_boxed.jpg"
                boxed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], boxed_filename)
                cv2.imwrite(boxed_filepath, processed_img)
                image_path = boxed_filepath

                if extracted.strip():
                    translated = translate_text(extracted, target_lang)
                else:
                    translated = "No text detected."

    return render_template("index.html",
                           extracted=extracted,
                           translated=translated,
                           image_path=image_path,
                           current_lang=target_lang)

# --- NEW API ROUTE FOR LIVE UPDATES ---
@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.json
    text = data.get("text", "")
    target_lang = data.get("target_lang", "fr")
    
    translated_text = translate_text(text, target_lang)
    
    return jsonify({"translated": translated_text})

if __name__ == "__main__":
    app.run(debug=True)