# 🎵 Sangeet Sathi — Hindi Music Recommendation System

A smart **Emotion-Based Hindi Song Recommender** that analyzes user mood (in English or Hinglish) and suggests songs accordingly.

---

## 🚀 Features

* 💬 Detects user emotion from text input
* 🤖 Machine Learning model (TF-IDF + Naive Bayes)
* 🎵 Recommends songs based on emotion
* 🖥️ Beautiful Tkinter GUI interface
* 🔄 Shuffle songs feature
* 📊 Emotion confidence & probability display
* 🌐 Supports Hinglish + English input

---

## 🧠 How It Works

1. User enters mood (e.g., *"Main bahut udaas hoon"*)
2. Text is preprocessed (cleaning, stopwords removal, stemming)
3. ML model predicts emotion
4. Songs are filtered from dataset
5. GUI displays recommendations

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Tkinter (GUI)

---

## 📂 Project Structure

```
Sangeet-Sathi/
│
├── sangeet_sathi_gui.py
├── hindi_songs_dataset.csv
└── README.md
```

---

## ▶️ How to Run

### 1. Clone Repository

```bash
git clone https://github.com/your-username/sangeet-sathi.git
cd sangeet-sathi
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn
```

### 3. Run the App

```bash
python sangeet_sathi_gui.py
```

---

## ⚠️ Important

* Keep the dataset file in the **same folder**:

```
hindi_songs_dataset.csv
```

---

## 📊 Dataset Requirements

CSV must contain these columns:

* song_title
* singer
* film
* year
* emotion
* keywords

---

## 🎯 Supported Emotions

* 😄 Happy
* 😢 Sad
* 💕 Romantic
* 😤 Angry
* 😌 Relaxed
* 💪 Motivational

---

## 🧪 Machine Learning Model

* TF-IDF Vectorizer (Feature Extraction)
* Multinomial Naive Bayes (Classification)
* Label Encoding
* Seed phrases + dataset training 

---

## 🖥️ GUI Features

* Clean modern interface
* Quick mood buttons
* Emotion confidence display
* Song table with details
* Shuffle recommendations

---

## 📸 Screenshots (Optional)

(Add screenshots here)

---

## 🤝 Contribution

Feel free to fork this repo and improve it!

---

## 📜 License

This project is open-source and free to use.

---

## 👨‍💻 Author

Vikky

---

⭐ If you like this project, don't forget to star the repo!

