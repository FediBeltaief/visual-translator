from flask import Flask, render_template, request
import os
import uuid
import easyocr
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'


ocr_reader = easyocr.Reader(['en'])  

model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


def translate_text(text):
    if all(ord(c) < 128 for c in text):
        pass

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

        if file:
            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_path = filepath

            result = ocr_reader.readtext(filepath)
            extracted_list = [item[1] for item in result]
            extracted = "\n".join(extracted_list)

            if extracted.strip():
                translated = translate_text(extracted)
            else:
                translated = "No text detected."

    return render_template("index.html",
                           extracted=extracted,
                           translated=translated,
                           image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)
