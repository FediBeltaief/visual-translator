import os
import csv
import json
import time
import numpy as np
from app import process_image_and_extract, translate_text


TEST_FOLDER = 'test_images'
os.makedirs(TEST_FOLDER, exist_ok=True)

TARGET_LANGS = ['fr', 'es', 'de']
BATCH_SIZES = [1, 2]

RESULTS_JSON = "grid_search_results.json"
RESULTS_CSV = "grid_search_results.csv"


grid_search_results = []  

test_images = [os.path.join(TEST_FOLDER, f) for f in os.listdir(TEST_FOLDER)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not test_images:
    print("No images found in test_images/ folder.")
    exit()

for lang in TARGET_LANGS:
    for batch_size in BATCH_SIZES:
        print(f"\nEvaluating: lang={lang}, batch_size={batch_size}")

        for img_path in test_images:
            start_time = time.time()

            img, ocr_results = process_image_and_extract(img_path)
            if not ocr_results:
                extracted_text = ""
            else:
                extracted_text = "\n".join([text for bbox, text in ocr_results])

            translated_text = translate_text(extracted_text, lang)

            processing_time = time.time() - start_time
            text_ratio = len(extracted_text) / max(1, len(translated_text))

            grid_search_results.append({
                'image': img_path,
                'target_lang': lang,
                'batch_size': batch_size,
                'processing_time': round(processing_time, 2),
                'extracted_text': extracted_text,
                'translated_text': translated_text,
                'text_ratio': round(text_ratio, 4)
            })

            print(f"Processed {img_path} | Extracted {len(extracted_text)} chars | Translated {len(translated_text)} chars | Time {processing_time:.2f}s")


with open(RESULTS_JSON, 'w', encoding='utf-8') as f_json:
    json.dump(grid_search_results, f_json, indent=4, ensure_ascii=False)

with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f_csv:
    fieldnames = ['image', 'target_lang', 'batch_size', 'processing_time', 'extracted_text', 'translated_text', 'text_ratio']
    writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(grid_search_results)

print(f"\nEvaluation completed. Results saved to {RESULTS_JSON} and {RESULTS_CSV}")
