"""
╔══════════════════════════════════════════════════════════════╗
║   🎵  Sangeet Sathi — Hindi Music Recommendation System      ║
║   Tech Stack: Python | Pandas | Scikit-learn | Tkinter       ║
║   Flow: User Input → Preprocess → Emotion ML → CSV Songs     ║
╚══════════════════════════════════════════════════════════════╝

USAGE:
    Keep  hindi_songs_dataset.csv  in the SAME folder as this file.
    Then run:  python sangeet_sathi_gui.py
"""

# ──────────────────────────────────────────────────────────────
# 0.  AUTO-INSTALL MISSING PACKAGES
# ──────────────────────────────────────────────────────────────
import sys, subprocess, importlib

for pkg in ["pandas", "numpy", "scikit-learn"]:
    mod = pkg.replace("-", "_").split("==")[0]
    if mod == "scikit_learn": mod = "sklearn"
    try:
        importlib.import_module(mod)
    except ImportError:
        print(f"Installing {pkg}…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

# ──────────────────────────────────────────────────────────────
# 1.  STANDARD IMPORTS
# ──────────────────────────────────────────────────────────────
import os, re, random, threading
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# ──────────────────────────────────────────────────────────────
# 2.  LOAD CSV
# ──────────────────────────────────────────────────────────────
CSV_NAME = "hindi_songs_dataset.csv"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, CSV_NAME)

if not os.path.exists(CSV_PATH):
    print(f"ERROR: '{CSV_NAME}' not found in {SCRIPT_DIR}")
    print("Place the CSV in the same folder as this script and try again.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)

# Validate required columns
REQUIRED_COLS = {"song_title", "singer", "film", "year", "emotion", "keywords"}
missing = REQUIRED_COLS - set(df.columns)
if missing:
    print(f"ERROR: CSV is missing columns: {missing}")
    sys.exit(1)

df["emotion"] = df["emotion"].str.strip().str.lower()
print(f"✓ Loaded {len(df)} songs | Emotions: {df['emotion'].value_counts().to_dict()}")

# ──────────────────────────────────────────────────────────────
# 3.  TEXT PREPROCESSING  (no NLTK needed)
# ──────────────────────────────────────────────────────────────
STOP_WORDS = {
    # English
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','he','him','his','himself','she','her','hers','herself','it','its',
    'itself','they','them','their','theirs','themselves','what','which','who','whom',
    'this','that','these','those','am','is','are','was','were','be','been','being',
    'have','has','had','having','do','does','did','doing','will','would','shall',
    'should','may','might','must','can','could','a','an','the','and','but','if','or',
    'because','as','until','while','of','at','by','for','with','about','against',
    'between','into','through','during','before','after','above','below','to','from',
    'up','down','in','out','on','off','over','under','again','further','then','once',
    'here','there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than','too',
    'very','just','now','get','got','feel','feeling','want','need','like','also','even',
    # Hinglish
    'hoon','hai','ho','main','mujhe','mera','meri','mere','aap','tum','yeh','woh',
    'kya','nahi','bahut','aur','ka','ki','ke','se','ko','par','mein','tha','thi',
    'raha','rahi','kar','karo','karta','karti','bhi','lekin','phir','jo','jab','tab',
    'ek','iss','us','unka','unki','unke','apna','apni','apne','sirf',
}

def simple_stem(word: str) -> str:
    """Lightweight suffix stripper (replaces PorterStemmer)."""
    for sfx in ['tion','ness','able','edly','ing','ful','less','ed','er','ly','est']:
        if len(word) > len(sfx) + 2 and word.endswith(sfx):
            return word[:-len(sfx)]
    return word

def preprocess(text: str) -> str:
    """Clean → tokenise → remove stopwords → stem."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [simple_stem(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)

# ──────────────────────────────────────────────────────────────
# 4.  BUILD TRAINING DATASET  (CSV keywords + seed phrases)
# ──────────────────────────────────────────────────────────────
SEED_PHRASES = {
    "happy": [
        "I am happy and excited today",
        "Main bahut khush hoon aaj",
        "feeling joyful wonderful celebration",
        "life is amazing great fun dance",
        "want to celebrate party upbeat cheerful",
        "bahut maza aa raha hai khushi",
    ],
    "sad": [
        "I am feeling sad and depressed",
        "Main bahut udaas hoon",
        "my heart is broken I want to cry",
        "feeling lonely empty missing someone",
        "dil bahut dard kar raha hai",
        "bahut rona aa raha hai udaas",
    ],
    "romantic": [
        "I am in love feeling romantic",
        "Mujhe pyaar ho gaya hai",
        "feeling romantic loving my heart beats",
        "ishq mein dooba hoon deeply in love",
        "tum se pyaar romantic feelings",
        "mohabbat romance dil loving",
    ],
    "angry": [
        "I am very angry and furious",
        "Main bahut gusse mein hoon",
        "feeling rage frustration everything makes me mad",
        "intense anger frustrated annoyed irritated",
        "bahut gussa aa raha hai krodh",
        "gussa gusse mein chidchida irritated rage",
        "krodh angry furious wrath temper losing",
    ],
    "relaxed": [
        "I want to relax and feel peaceful",
        "Mujhe sukoon chahiye calm serene",
        "feeling calm chill unwind tranquil",
        "need soothing gentle music meditate",
        "shanti peace aram sukoon",
    ],
    "motivational": [
        "I need motivation and inspiration to achieve",
        "Mujhe motivation chahiye dreams goals",
        "never give up I will succeed determined",
        "feeling determined to win accomplish",
        "himmat hausla inspire achieve success",
    ],
}

texts, labels = [], []

# From CSV  (keywords column is the primary signal)
for _, row in df.iterrows():
    texts.append(str(row["keywords"]))
    labels.append(row["emotion"])

# From seed phrases
for emo, phrases in SEED_PHRASES.items():
    for p in phrases:
        texts.append(p)
        labels.append(emo)

train_df = pd.DataFrame({"text": texts, "emotion": labels})
train_df["processed"] = train_df["text"].apply(preprocess)

# ──────────────────────────────────────────────────────────────
# 5.  TRAIN TF-IDF + NAIVE BAYES MODEL
# ──────────────────────────────────────────────────────────────
le = LabelEncoder()
X  = train_df["processed"]
y  = le.fit_transform(train_df["emotion"])

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
    ("clf",   MultinomialNB(alpha=0.5)),
])
model.fit(X, y)
print("✓ Model trained")

# ──────────────────────────────────────────────────────────────
# 6.  HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────
EMOJI   = {"happy": "😄", "sad": "😢", "romantic": "💕",
           "angry": "😤", "relaxed": "😌", "motivational": "💪"}
COLORS  = {"happy": "#F59E0B", "sad": "#3B82F6", "romantic": "#EC4899",
           "angry": "#EF4444", "relaxed": "#22C55E", "motivational": "#6366F1"}


def predict_emotion(user_text: str):
    """Returns (emotion_str, confidence_float, all_scores_dict)."""
    processed  = preprocess(user_text)
    probs      = model.predict_proba([processed])[0]
    pred_idx   = int(np.argmax(probs))
    emotion    = le.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])
    all_scores = {
        le.inverse_transform([i])[0]: round(p * 100, 1)
        for i, p in enumerate(probs)
    }
    return emotion, confidence, all_scores


def get_songs(emotion: str, top_n: int = 5) -> pd.DataFrame:
    """Fetch a random sample of songs from the CSV for the detected emotion."""
    pool = df[df["emotion"] == emotion].copy()
    n    = min(top_n, len(pool))
    return pool.sample(n=n, random_state=random.randint(0, 99999))\
               .reset_index(drop=True)[["song_title", "singer", "film", "year"]]


# ──────────────────────────────────────────────────────────────
# 7.  TKINTER GUI
# ──────────────────────────────────────────────────────────────
class SangeetSathi(tk.Tk):
    # ── Design tokens ──────────────────────────────────────────
    BG       = "#0F0F1A"
    SURFACE  = "#1A1A2E"
    CARD     = "#16213E"
    ACCENT   = "#7C3AED"
    TEXT     = "#F1F5F9"
    MUTED    = "#94A3B8"
    BORDER   = "#2D2D4E"
    F_TITLE  = ("Georgia", 22, "bold")
    F_H2     = ("Georgia", 13, "bold")
    F_BODY   = ("Segoe UI", 11)
    F_SMALL  = ("Segoe UI", 9)
    F_MONO   = ("Consolas", 10)
    F_BTN    = ("Segoe UI", 12, "bold")

    def __init__(self):
        super().__init__()
        self.title("🎵 Sangeet Sathi — Hindi Music Recommender")
        self.geometry("880x720")
        self.minsize(740, 580)
        self.configure(bg=self.BG)
        self._build_ui()
        self._center()

    # ── Layout ─────────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()
        self._build_input_panel()
        tk.Frame(self, bg=self.BORDER, height=1).pack(fill="x", padx=20)
        self._build_results_panel()

    def _build_header(self):
        hdr = tk.Frame(self, bg="#0D0B1F", pady=16)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🎵  Sangeet Sathi",
                 font=self.F_TITLE, bg="#0D0B1F", fg="#A78BFA").pack()
        tk.Label(hdr,
                 text=f"Emotion-Powered Hindi Music Recommendations  •  {len(df)} songs",
                 font=self.F_SMALL, bg="#0D0B1F", fg=self.MUTED).pack(pady=(2, 0))

    def _build_input_panel(self):
        pnl = tk.Frame(self, bg=self.BG, pady=14, padx=24)
        pnl.pack(fill="x")

        tk.Label(pnl, text="💬  How are you feeling right now?",
                 font=self.F_H2, bg=self.BG, fg=self.TEXT).pack(anchor="w")
        tk.Label(pnl, text="Type in English or Hinglish, then press Enter",
                 font=self.F_SMALL, bg=self.BG, fg=self.MUTED).pack(anchor="w", pady=(2, 8))

        # Entry box
        ef = tk.Frame(pnl, bg=self.ACCENT, bd=1)
        ef.pack(fill="x")
        inner = tk.Frame(ef, bg=self.CARD)
        inner.pack(fill="x", padx=1, pady=1)
        self.entry = tk.Entry(inner, font=("Segoe UI", 13),
                              bg=self.CARD, fg=self.TEXT,
                              insertbackground=self.TEXT,
                              relief="flat", bd=10)
        self.entry.pack(fill="x", ipady=6)
        self.entry.bind("<Return>", lambda _: self._on_recommend())
        self.entry.focus()

        # Quick-mood buttons
        quick = tk.Frame(pnl, bg=self.BG, pady=8)
        quick.pack(fill="x")
        tk.Label(quick, text="Quick mood:", font=self.F_SMALL,
                 bg=self.BG, fg=self.MUTED).pack(side="left", padx=(0, 6))

        quick_moods = [
            ("😄 Happy",      "Main bahut khush hoon aaj, celebrate karna hai"),
            ("😢 Sad",        "Main bahut udaas hoon, dil bahut dard kar raha hai"),
            ("💕 Romantic",   "Mujhe pyaar ho gaya hai, romantic feel ho raha hai"),
            ("😤 Angry",      "Main bahut gusse mein hoon, bahut frustrated hoon"),
            ("😌 Relaxed",    "Mujhe sukoon chahiye, aram karna hai shanti se"),
            ("💪 Motivated",  "I need motivation to achieve my dreams and goals"),
        ]
        for label, phrase in quick_moods:
            tk.Button(quick, text=label, font=self.F_SMALL,
                      bg=self.SURFACE, fg=self.TEXT,
                      activebackground=self.ACCENT, activeforeground="white",
                      relief="flat", bd=0, padx=9, pady=5, cursor="hand2",
                      command=lambda p=phrase: self._fill(p)
                      ).pack(side="left", padx=3)

        # Recommend button + status
        row = tk.Frame(pnl, bg=self.BG, pady=6)
        row.pack(fill="x")
        self.btn = tk.Button(row, text="🎵  Get Recommendations",
                             font=self.F_BTN, bg=self.ACCENT, fg="white",
                             activebackground="#6D28D9", activeforeground="white",
                             relief="flat", bd=0, padx=28, pady=10, cursor="hand2",
                             command=self._on_recommend)
        self.btn.pack(side="left")
        self.status = tk.Label(row, text="", font=self.F_SMALL,
                               bg=self.BG, fg=self.MUTED)
        self.status.pack(side="left", padx=14)

    def _build_results_panel(self):
        self.res = tk.Frame(self, bg=self.BG)
        self.res.pack(fill="both", expand=True, padx=24, pady=12)
        self._placeholder()

    # ── Helpers ────────────────────────────────────────────────
    def _center(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _fill(self, text):
        self.entry.delete(0, "end")
        self.entry.insert(0, text)
        self.entry.focus()

    def _clear(self):
        for w in self.res.winfo_children():
            w.destroy()

    def _placeholder(self):
        self._clear()
        tk.Label(self.res,
                 text="✨  Enter your mood above and press Enter",
                 font=("Segoe UI", 13), bg=self.BG, fg=self.BORDER
                 ).pack(expand=True)

    # ── Recommend flow ─────────────────────────────────────────
    def _on_recommend(self):
        text = self.entry.get().strip()
        if not text:
            messagebox.showwarning("Empty Input",
                                   "Please describe how you're feeling!")
            return
        self.btn.config(state="disabled")
        self.status.config(text="Detecting emotion…")
        threading.Thread(target=self._run, args=(text,), daemon=True).start()

    def _run(self, text):
        emotion, confidence, scores = predict_emotion(text)
        songs = get_songs(emotion)
        self.after(0, lambda: self._display(text, emotion, confidence, scores, songs))

    def _display(self, text, emotion, confidence, scores, songs):
        self._clear()
        color = COLORS.get(emotion, self.ACCENT)
        emoji = EMOJI.get(emotion, "🎵")

        # ── Emotion banner ──
        banner = tk.Frame(self.res, bg=color, pady=12, padx=18)
        banner.pack(fill="x", pady=(0, 10))

        left = tk.Frame(banner, bg=color)
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text=f"{emoji}  {emotion.upper()}",
                 font=("Segoe UI", 20, "bold"), bg=color, fg="white").pack(anchor="w")
        tk.Label(left, text=f'You said: "{text}"',
                 font=("Segoe UI", 9, "italic"), bg=color, fg="white").pack(anchor="w", pady=(2, 0))

        right = tk.Frame(banner, bg=color)
        right.pack(side="right", padx=12)
        tk.Label(right, text=f"{confidence:.0%}",
                 font=("Segoe UI", 28, "bold"), bg=color, fg="white").pack()
        tk.Label(right, text="confidence",
                 font=("Segoe UI", 8), bg=color, fg="white").pack()

        # ── Probability bars ──
        bar_card = tk.Frame(self.res, bg=self.SURFACE, pady=10, padx=14)
        bar_card.pack(fill="x", pady=(0, 10))
        tk.Label(bar_card, text="Emotion Probabilities",
                 font=("Segoe UI", 9, "bold"), bg=self.SURFACE, fg=self.MUTED
                 ).pack(anchor="w", pady=(0, 6))

        for emo, score in sorted(scores.items(), key=lambda x: -x[1]):
            row = tk.Frame(bar_card, bg=self.SURFACE)
            row.pack(fill="x", pady=2)

            em_lbl = f"{EMOJI.get(emo,'')} {emo}"
            tk.Label(row, text=f"{em_lbl:<18}", font=self.F_MONO,
                     bg=self.SURFACE, fg=self.MUTED,
                     width=18, anchor="w").pack(side="left")

            track = tk.Frame(row, bg=self.BG, height=12, width=270)
            track.pack(side="left", padx=6)
            track.pack_propagate(False)
            bar_w = max(2, int(score * 2.7))
            tk.Frame(track, bg=COLORS.get(emo, self.ACCENT),
                     height=12, width=bar_w).place(x=0, y=0, relheight=1)

            tk.Label(row, text=f"{score}%", font=self.F_MONO,
                     bg=self.SURFACE, fg=self.TEXT).pack(side="left")

        # ── Song recommendations ──
        tk.Label(self.res,
                 text=f"🎵  Recommended Songs — {emotion.capitalize()} Mood",
                 font=self.F_H2, bg=self.BG, fg=self.TEXT
                 ).pack(anchor="w", pady=(8, 4))
        tk.Label(self.res,
                 text=f"Showing {len(songs)} of {len(df[df['emotion']==emotion])} songs in this category  •  Shuffle for more",
                 font=self.F_SMALL, bg=self.BG, fg=self.MUTED
                 ).pack(anchor="w", pady=(0, 6))

        # Style Treeview
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("S.Treeview",
                        background=self.CARD,
                        fieldbackground=self.CARD,
                        foreground=self.TEXT,
                        rowheight=34,
                        font=("Segoe UI", 10))
        style.configure("S.Treeview.Heading",
                        background=self.SURFACE,
                        foreground=color,
                        font=("Segoe UI", 10, "bold"),
                        relief="flat")
        style.map("S.Treeview",
                  background=[("selected", self.ACCENT)],
                  foreground=[("selected", "white")])

        cols = ("#", "Song Title", "Singer", "Film", "Year")
        tree = ttk.Treeview(self.res, columns=cols, show="headings",
                            height=len(songs), style="S.Treeview",
                            selectmode="browse")
        widths = [30, 260, 160, 190, 55]
        for col, w in zip(cols, widths):
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor="w" if w > 60 else "center")

        for i, (_, row) in enumerate(songs.iterrows()):
            tag = "even" if i % 2 == 0 else "odd"
            tree.insert("", "end",
                        values=(i+1, row["song_title"], row["singer"],
                                row["film"], int(row["year"])),
                        tags=(tag,))
        tree.tag_configure("even", background=self.CARD)
        tree.tag_configure("odd",  background=self.SURFACE)
        tree.pack(fill="x")

        # ── Action buttons ──
        btn_row = tk.Frame(self.res, bg=self.BG, pady=10)
        btn_row.pack(fill="x")

        def shuffle():
            new_songs = get_songs(emotion)
            self._display(text, emotion, confidence, scores, new_songs)

        tk.Button(btn_row, text="🔀  Shuffle Songs",
                  font=self.F_SMALL, bg=self.SURFACE, fg=self.TEXT,
                  activebackground=self.ACCENT, activeforeground="white",
                  relief="flat", bd=0, padx=14, pady=6, cursor="hand2",
                  command=shuffle).pack(side="left", padx=(0, 8))

        tk.Button(btn_row, text="🔄  Try Another Mood",
                  font=self.F_SMALL, bg=self.SURFACE, fg=self.MUTED,
                  activebackground=self.BORDER, relief="flat",
                  bd=0, padx=14, pady=6, cursor="hand2",
                  command=self._reset).pack(side="left")

        self.btn.config(state="normal")
        self.status.config(text="")

    def _reset(self):
        self.entry.delete(0, "end")
        self._placeholder()
        self.entry.focus()


# ──────────────────────────────────────────────────────────────
# 8.  ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SangeetSathi()
    app.mainloop()
