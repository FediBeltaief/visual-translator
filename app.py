from flask import Flask, render_template, request
import os
import uuid
import easyocr
from transformers import MarianMTModel, MarianTokenizer
import cv2
from PIL import Image
import numpy as np  


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ocr_reader = easyocr.Reader(['en'])

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
    tokenizer, model = get_model(lang)

    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translation = model.generate(**tokens)
    return tokenizer.decode(translation[0], skip_special_tokens=True)


@app.route("/", methods=["GET", "POST"])
def index():
    translated = None
    extracted = None
    image_path = None

    if request.method == "POST":
        file = request.files['image']
        target_lang = request.form.get("target_lang", "fr") 

        if file:
            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result = ocr_reader.readtext(filepath)
            extracted_list = [item[1] for item in result]
            extracted = "\n".join(extracted_list)
            image_cv = cv2.imread(filepath)
            for (bbox, text, prob) in result:
                pts = [(int(x), int(y)) for x, y in bbox]
                cv2.polylines(image_cv, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)

            boxed_filename = f"{uuid.uuid4()}_boxed.jpg"
            boxed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], boxed_filename)
            cv2.imwrite(boxed_filepath, image_cv)

            image_path = boxed_filepath

            if extracted.strip():
                translated = translate_text(extracted, target_lang)
            else:
                translated = "No text detected."

    return render_template("index.html",
                           extracted=extracted,
                           translated=translated,
                           image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)
