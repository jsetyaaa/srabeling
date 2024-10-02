from flask import Flask, render_template, request, send_file, redirect, url_for
from google_play_scraper import reviews
import pandas as pd
import re
from joblib import load
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

# Load model SVM dan CountVectorizer
model_path = "model/svm_model.pkl"
vectorizer_path = "model/count_vectorizer.pkl"
model = load(model_path)
vectorizer = load(vectorizer_path)

# Inisialisasi Stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi Preprocessing
def preprocess_text(text):
    # Tahap 1: Normalisasi (membersihkan tanda baca)
    text_clean = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tahap 2: Normalisasi kata-kata yang sering disingkat atau salah
    text_normalized = text_clean
    text_normalized = re.sub(r'\bgk\b|\bg\b|\bngk\b|\bgak\b|\bngak\b|\bnggak\b|\bgx\b|\btdk\b|\bga\b', 'tidak', text_normalized)
    text_normalized = re.sub(r'\bgabisa\b', 'tidak bisa', text_normalized)
    text_normalized = re.sub(r'\btp\b|\btapi\b', 'namun', text_normalized)
    text_normalized = re.sub(r'\bok\b|\boke\b|\bokee\b|\bokk\b', 'baik', text_normalized)
    text_normalized = re.sub(r'\bsdh\b|\budh\b|\budah\b', 'sudah', text_normalized)
    text_normalized = re.sub(r'\bapk\b', 'aplikasi', text_normalized)
    text_normalized = re.sub(r'\beror\b', 'error', text_normalized)
    text_normalized = re.sub(r'\bbs\b|\bbsa\b', 'bisa', text_normalized)
    text_normalized = re.sub(r'\bdpt\b|\bdpat\b|\bdapet\b', 'dapat', text_normalized)
    text_normalized = re.sub(r'\byg\b', 'yang', text_normalized)
    text_normalized = re.sub(r'\bjgn\b|\bjngn\b|\bjangn\b', 'jangan', text_normalized)
    text_normalized = re.sub(r'\bblm\b|\bblum\b', 'belum', text_normalized)
    text_normalized = re.sub(r'\bkrn\b|\bkrna\b', 'karena', text_normalized)
    text_normalized = re.sub(r'\bjos\b', 'sangat baik', text_normalized)
    text_normalized = re.sub(r'\butk\b|\buntk\b', 'untuk', text_normalized)
    
    # Tahap 3: Case Folding (mengubah teks menjadi huruf kecil)
    text_lower = text_normalized.lower()
    
    # Tahap 4: Tokenisasi (memisahkan teks menjadi kata-kata individu)
    tokens = text_lower.split()
    
    # Tahap 5: Stemming menggunakan Sastrawi
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Gabungkan kembali menjadi satu string setelah stemming
    preprocessed_text = ' '.join(stemmed_tokens)
    
    return {
        'cleaned_text': text_clean,
        'normalized_text': text_normalized,
        'lower_text': text_lower,
        'tokens': tokens,
        'stemmed_tokens': stemmed_tokens,
        'preprocessed_text': preprocessed_text
    }

# Halaman Utama
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint untuk scraping
@app.route('/scrape', methods=['POST'])
def scrape():
    app_id = request.form.get('app_id')
    comment_limit = int(request.form.get('limit'))

    # Scraping ulasan dari Google Play
    scraped_reviews, _ = reviews(app_id, lang='id', count=comment_limit)
    
    # Simpan hasil ke dataframe
    comments_df = pd.DataFrame(scraped_reviews)[['userName', 'content', 'score', 'at']]
    comments_df['content'] = comments_df['content'].apply(str)

    # Preprocessing setiap konten ulasan
    preprocessing_results = comments_df['content'].apply(preprocess_text)

    # Membuat kolom untuk hasil preprocessing
    comments_df['cleaned_content'] = preprocessing_results.apply(lambda x: x['cleaned_text'])
    comments_df['normalized_content'] = preprocessing_results.apply(lambda x: x['normalized_text'])
    comments_df['lower_content'] = preprocessing_results.apply(lambda x: x['lower_text'])
    comments_df['tokens'] = preprocessing_results.apply(lambda x: x['tokens'])
    comments_df['stemmed_content'] = preprocessing_results.apply(lambda x: x['preprocessed_text'])

    # Simpan hasil scraping ke file CSV
    csv_filename = 'scraped_comments.csv'
    comments_df.to_csv(csv_filename, index=False)

    # Redirect ke halaman dengan opsi download atau klasifikasi
    return redirect(url_for('result', file=csv_filename))

# Halaman hasil scraping
@app.route('/result')
def result():
    file = request.args.get('file')
    return render_template('result.html', file=file)

# Download hasil scraping
@app.route('/download/<filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

# Klasifikasi hasil scraping
@app.route('/classify/<filename>', methods=['POST'])
def classify(filename):
    # Baca file scraping dari CSV
    comments_df = pd.read_csv(filename)

    # Pastikan kolom 'stemmed_content' sudah ada dari preprocessing sebelumnya
    if 'stemmed_content' not in comments_df.columns:
        return "Kolom 'stemmed_content' tidak ditemukan. Pastikan preprocessing sudah dilakukan.", 400

    # Mengganti nilai NaN di kolom 'stemmed_content' dengan string kosong
    comments_df['stemmed_content'] = comments_df['stemmed_content'].fillna('')

    # Transformasi teks hasil stemming menggunakan CountVectorizer
    comments_transformed = vectorizer.transform(comments_df['stemmed_content'])

    # Prediksi label menggunakan model SVM
    comments_df['label'] = model.predict(comments_transformed)

    # Simpan hasil klasifikasi ke file CSV baru
    classified_filename = 'classified_comments.csv'
    comments_df.to_csv(classified_filename, index=False)

    # Redirect ke halaman hasil dan kirim parameter classified=True
    return redirect(url_for('result', file=classified_filename, classified=True))




if __name__ == '__main__':
    app.run(debug=True)
