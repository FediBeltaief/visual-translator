## Visual Translator: Image to Text to Another Language

This project is a super simple web app that helps you translate text from images into another language! It's like having a pocket translator for signs, documents, or anything with text you don't understand.

It's built using Flask (to run the website), EasyOCR (to read the text in the image), and MarianMT (to translate the text).

### How it Works

The goal is to make translating text from images quick and easy. Here's how it works:

1.  **Upload an Image:** First, you upload a picture to the website.
2.  **Text Extraction (OCR):** The app uses OCR to find and grab any text in the image.
3.  **Automatic Translation:** The text is then automatically translated into the language you want.
4.  **See the Results:** Finally, you'll see the original image with boxes around the detected text, and the translated text below.

The interface:

![User Interface](https://i.imgur.com/pgeVOPs.png)

### Cool Features

1.  **Easy Image Upload:**

    There's a simple form where you can easily upload any image from your computer.

2.  **Smart Text Detection (OCR):**

    The app uses EasyOCR to find English text in the image. The text it finds is highlighted with boxes, so you can see what the app detected.

    Example:

    ![Detected Text Example](https://i.imgur.com/lClVOqb.png)

3.  **Machine Translation Magic:**

    The app uses a special model called `Helsinki-NLP/opus-mt-en-fr` to translate the text. By default, it translates from English to French. But you can change the model to translate between other languages, too!

4.  **Clean and Simple Interface:**

    The website shows you everything you need in a clear and easy way:

    *   The image you uploaded.
    *   The text that was detected in the image.
    *   The translated text.

### Project Structure

Here's how the project files are organized:

```
project/
│
├── app.py        # The main file that runs the web app
├── templates/    # Contains the HTML files for the website
│   └── index.html  # The main page of the web app
├── static/       # Holds static files like images
│   └── uploads/   # Where uploaded images are stored
└── README.md     # This file!
```

**Main Operations**

This app handles:

*   File uploads
*   OCR (detecting text in images)
*   Drawing boxes around detected text
*   Translation
*   Showing the results

**Customization**

You can change the app to:

*   Support more languages by using different OCR and translation models.
*   Make the website look nicer and easier to use.
*   Add support for using your phone's camera.
*   Detect the language of the text automatically.

**Troubleshooting**

*   **GPU Not Used:**

    EasyOCR and MarianMT usually use your computer's CPU. If you have a good graphics card (GPU) and installed PyTorch with CUDA, the app will automatically use it.
*   **Large Image Slow:**

    If the app is slow with large images, try making the image smaller before uploading it.

**License**

This project is released under the MIT License.