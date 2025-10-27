import tkinter as tk
from tkinter import ttk, font
from tkinter.scrolledtext import ScrolledText
import time
from collections import Counter, defaultdict
import string
import math
import statistics

class TypingCoachApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Typhora")
        self.root.geometry("980x680")

        # ---------- Modern, Pastel Color Palette (like CSS variables) ----------
        self.COLOR_BG = "#F0F4F8"          # Light, airy blue-gray
        self.COLOR_TEXT = "#34495E"        # Dark slate blue (softer than black)
        self.COLOR_PRIMARY = "#76D7C4"     # Mint green for buttons
        self.COLOR_PRIMARY_ACTIVE = "#60B9A7" # Darker mint for button press
        self.COLOR_SECONDARY = "#A9CCE3"    # Soft blue for accents/frames
        self.COLOR_ACCENT = "#D7BDE2"       # Pastel lavender for highlights
        self.COLOR_CORRECT = "#2ECC71"     # Vibrant, clear green
        self.COLOR_WRONG = "#E74C3C"       # Clear, soft red
        self.COLOR_WHITE = "#FFFFFF"

        self.root.configure(bg=self.COLOR_BG)

        # ---------- Font Scheme ----------
        self.FONT_HEADING = font.Font(family="Segoe UI", size=28, weight="bold")
        self.FONT_SUBHEADING = font.Font(family="Segoe UI", size=22, weight="bold")
        self.FONT_BODY_BOLD = font.Font(family="Segoe UI", size=16, weight="bold")
        self.FONT_BODY = font.Font(family="Segoe UI", size=16)
        self.FONT_BUTTON = font.Font(family="Segoe UI", size=14, weight="bold")
        self.FONT_COMBO = font.Font(family="Segoe UI", size=14)
        self.FONT_SMALL = font.Font(family="Segoe UI", size=12)
        self.FONT_FEEDBACK = font.Font(family="Consolas", size=12)

        # ---------- Style for TTK Widgets (Dropdowns) ----------
        style = ttk.Style()
        style.theme_use('clam') # A clean, modern theme base
        style.configure("TCombobox",
                        selectbackground=self.COLOR_WHITE,
                        selectforeground=self.COLOR_TEXT,
                        fieldbackground=self.COLOR_WHITE,
                        background=self.COLOR_PRIMARY,
                        foreground=self.COLOR_TEXT,
                        arrowcolor=self.COLOR_TEXT,
                        font=self.FONT_COMBO)
        # Fix for dropdown arrow hover color
        style.map('TCombobox', fieldbackground=[('readonly','white')])
        style.map('TCombobox', selectbackground=[('readonly', 'white')])
        style.map('TCombobox', selectforeground=[('readonly', self.COLOR_TEXT)])

        # ---------------- Sentences bank (kept EXACTLY as provided) ----------------
        self.sentences = {
            "Beginner": [
                """Artificial intelligence represents the frontier of computer science, dedicated to creating systems capable of performing tasks that normally require human intelligence. The fundamental pursuit of AI is to build machines that can reason, learn, and act autonomously, profoundly transforming industries and our daily lives. Early concepts of artificial intelligence were rooted in symbolic logic and rule-based systems, where knowledge was explicitly encoded by programmers to solve well-defined problems. The famous Turing Test, proposed by Alan Turing, offered a benchmark for machine intelligence, questioning whether a machine could exhibit intelligent behavior indistinguishable from that of a human.""",
                """The computer is a very useful tool. We use it every day to work and play. The internet connects computers all over the world. You can send an email to a friend in another country. You can also watch videos or listen to music online. Websites give us news and a lot of information. It is important to keep your computer safe from a virus. A good password helps protect your files and personal data. Learning to code helps you build your own apps and games. The future of technology is bright and full of new ideas.""",
                """Artificial intelligence helps make machines smart. A smart machine can learn new things from data. Robots are machines that can do physical tasks. Some robots work in a large factory to build cars. Other robots can help clean your house or mow the lawn. Voice assistants like Siri or Alexa use AI to understand what you say. They can answer your questions or play your favorite song. AI is also used in games to make the characters act in smart ways. The goal is to make computers that can think and solve problems just like people do.""",
            ],
            "Intermediate": [
                """Machine learning algorithms are engineered to facilitate predictive analysis and pattern recognition without explicit programming. Supervised learning, for instance, requires a meticulously labeled dataset to train a model for classification or regression tasks, discerning relationships between input variables and a target output. Conversely, unsupervised learning explores unlabeled data to discover inherent structures, employing techniques like clustering to group similar data points together. The process of feature engineering is exceptionally influential, as the quality of the input data fundamentally determines the performance and accuracy of the resulting predictive model. Overfitting is a persistent challenge where a model memorizes the training data idiosyncratically, consequently failing to generalize to new, unfamiliar information.""",
                """Cybersecurity professionals implement sophisticated measures to protect digital infrastructure from malicious intrusions and unauthorized access. Comprehensive security protocols often involve multi-factor authentication, which requires multiple forms of verification before granting a user entry into a system. Encryption is another fundamental component, scrambling data into an unreadable format to prevent interception by nefarious actors during transmission. Organizations must remain perpetually vigilant against evolving threats like phishing schemes, ransomware, and zero-day vulnerabilities. Proactive threat intelligence and continuous network monitoring are indispensable for identifying and neutralizing potential security breaches before they can cause catastrophic damage to an organization's reputation and financial stability.""",
                """The architecture of a deep neural network is characterized by its numerous interconnected layers, which enables the hierarchical learning of complex features from raw input. Convolutional Neural Networks, or CNNs, are exceptionally proficient at processing spatial data, making them quintessential for image recognition and computer vision applications. Recurrent Neural Networks (RNNs) are specifically designed to manage sequential data, demonstrating remarkable capabilities in natural language processing and time-series forecasting. The quintessential training mechanism, known as backpropagation, iteratively adjusts the network's parameters to minimize the discrepancy between its predictions and the actual outcomes. Optimizing these immensely complex systems requires substantial computational resources and a nuanced understanding of hyperparameter tuning.""",
            ],
            "Advanced": [
                """To deploy the new service, first, SSH into the server at 192.168.1.101 using the private key deploy_key_rsa. Navigate to /var/www/project_alpha/ and execute git pull origin main to fetch the latest updates from the repository. The configuration file, config.yml, requires a new database entry: DB_HOST=db-main-v2.eu-west-1.rds.amazonaws.com. Ensure that port 5432 is open and not blocked by the ufw firewall. The system's process ID (PID) can be found by running ps aux | grep 'gunicorn'. The final build artifact, app_build_v3.4.1-beta.tar.gz, has a checksum (SHA-256) of a1b2c3d4-e5f6-7890-abcd-ef1234567890 for verification.""",
                """The SQL query SELECT user_id, email, last_login FROM users WHERE account_status = 'active' AND signup_date > '2024-01-01'; should be optimized by adding an index on the signup_date column. A common regular expression ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ is used for validating email formats. In Python, you might see a dictionary comprehension like {k: v**2 for (k, v) in some_dict.items() if v > 100}. To grant execute permissions to a shell script named run_backup.sh, you would use the command chmod +x run_backup.sh. The JSON response from the API endpoint /api/v2/products?id=4815162342 might look like {"product_id": 4815162342, "stock_level": 500, "price": 99.99}."""
            ]
        }
        
        # ---------------- Tracking (session-wide) ----------------
        self.time_limit = 30
        self.start_time = None
        self.correct_words = 0
        self.total_words = 0
        self.sentence_index = 0
        self.current_sentence = ""
        self.total_keystrokes = 0
        self.correct_keystrokes = 0
        self.key_hits = Counter()
        self.key_mistakes = Counter()
        self.row_mistakes = Counter()
        self.history = []

        # --- NEW: progress history across sessions for predictive analysis ---
        self.progress_history = []  # list of dicts: {"wpm": int, "accuracy": int, "cpm": int, "timestamp": float}

        # --- NEW: keystroke-level logging for rhythm/pacing ---
        self.keystroke_log = []  # list of dicts: {"t": float, "typed": str, "expected": str|None, "correct": bool}

        # --- NEW: keyboard adjacency map for adjacent-key error analysis ---
        self.adjacent_map = self.build_qwerty_adjacency()

        self.show_start_page()

    def create_styled_button(self, parent, text, command, font=None, width=18):
        """Helper to create consistently styled buttons."""
        if font is None:
            font = self.FONT_BUTTON
            
        return tk.Button(parent, text=text, command=command, font=font, width=width,
                         bg=self.COLOR_PRIMARY, fg=self.COLOR_WHITE,
                         activebackground=self.COLOR_PRIMARY_ACTIVE,
                         activeforeground=self.COLOR_WHITE,
                         relief="flat", borderwidth=0, pady=5)

    # ---------- PAGE 1 ----------
    def show_start_page(self):
        self.clear_window()
        main_frame = tk.Frame(self.root, bg=self.COLOR_BG)
        main_frame.pack(expand=True)
        
        tk.Label(main_frame, text="Typhora", font=self.FONT_HEADING, 
                 bg=self.COLOR_BG, fg=self.COLOR_TEXT).pack(pady=(0, 20))
        
        self.create_styled_button(main_frame, "Start", self.show_selection_page).pack(pady=8)
        self.create_styled_button(main_frame, "Exit", self.root.quit).pack()

    # ---------- PAGE 2 ----------
    def show_selection_page(self):
        self.clear_window()
        tk.Label(self.root, text="Select Test Settings", font=self.FONT_SUBHEADING, 
                 bg=self.COLOR_BG, fg=self.COLOR_TEXT).pack(pady=(40, 30))

        # Use a frame for better alignment and padding
        settings_frame = tk.Frame(self.root, bg=self.COLOR_BG, pady=10)
        settings_frame.pack()
        
        tk.Label(settings_frame, text="Difficulty:", font=self.FONT_BODY, bg=self.COLOR_BG, fg=self.COLOR_TEXT).grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.diff_var = tk.StringVar()
        diff_dropdown = ttk.Combobox(settings_frame, textvariable=self.diff_var, state="readonly",
                                      values=["Beginner", "Intermediate", "Advanced"], font=self.FONT_COMBO, width=22)
        diff_dropdown.grid(row=0, column=1, padx=10, pady=10)
        diff_dropdown.current(0)

        tk.Label(settings_frame, text="Duration:", font=self.FONT_BODY, bg=self.COLOR_BG, fg=self.COLOR_TEXT).grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.time_var = tk.StringVar()
        time_dropdown = ttk.Combobox(settings_frame, textvariable=self.time_var, state="readonly",
                                      values=["30 seconds", "1 minute"], font=self.FONT_COMBO, width=22)
        time_dropdown.grid(row=1, column=1, padx=10, pady=10)
        time_dropdown.current(0)

        btns_frame = tk.Frame(self.root, bg=self.COLOR_BG)
        btns_frame.pack(pady=30)
        self.create_styled_button(btns_frame, "Begin Test", self.start_test).grid(row=0, column=0, padx=10)
        self.create_styled_button(btns_frame, "Back", self.show_start_page, font=self.FONT_SMALL, width=12).grid(row=0, column=1, padx=10)

    # ---------- PAGE 3 (Test) ----------
    def start_test(self):
        self.clear_window()

        self.total_keystrokes = 0
        self.correct_keystrokes = 0
        self.key_hits.clear()
        self.key_mistakes.clear()
        self.row_mistakes.clear()
        self.history.clear()
        self.keystroke_log.clear()

        difficulty = self.diff_var.get()
        self.sent_list = self.sentences[difficulty][:]
        self.sentence_index = 0

        self.time_limit = 30 if self.time_var.get() == "30 seconds" else 60
        self.start_time = time.time()
        self.correct_words = 0
        self.total_words = 0

        header = tk.Frame(self.root, bg=self.COLOR_BG)
        header.pack(pady=(10, 0), fill="x", padx=20)
        tk.Label(header, text=f"Difficulty: {difficulty}", font=self.FONT_BODY, bg=self.COLOR_BG, fg=self.COLOR_TEXT).pack(side="left")
        self.timer_label = tk.Label(header, text=f"Time Left: {self.time_limit}s", font=self.FONT_BODY_BOLD, bg=self.COLOR_BG, fg=self.COLOR_ACCENT)
        self.timer_label.pack(side="right")

        self.sentence_display = ScrolledText(self.root, font=("Segoe UI", 18), height=6, wrap="word",
                                             bg=self.COLOR_WHITE, fg=self.COLOR_TEXT, padx=12, pady=12,
                                             relief="flat", borderwidth=2)
        self.sentence_display.pack(padx=20, pady=18, fill="both", expand=True)
        self.sentence_display.config(state="disabled")

        self.entry = tk.Entry(self.root, font=("Segoe UI", 16), bg=self.COLOR_WHITE, fg=self.COLOR_TEXT,
                              insertbackground=self.COLOR_TEXT, relief="flat", borderwidth=4)
        self.entry.pack(pady=(0, 10), fill="x", padx=20)
        self.entry.focus_set()

        ctrls = tk.Frame(self.root, bg=self.COLOR_BG)
        ctrls.pack(pady=10)
        self.create_styled_button(ctrls, "View Feedback", self.show_feedback_page, font=self.FONT_SMALL, width=16).grid(row=0, column=0, padx=8)
        self.create_styled_button(ctrls, "Back to Selection", self.show_selection_page, font=self.FONT_SMALL, width=16).grid(row=0, column=1, padx=8)

        self.entry.bind("<Return>", self.submit_sentence)
        self.entry.bind("<KeyRelease>", self.on_key_release)

        self.sentence_display.tag_config("correct", foreground=self.COLOR_CORRECT)
        self.sentence_display.tag_config("wrong", foreground=self.COLOR_WRONG)

        self.next_sentence()
        self.update_timer()

    def keyboard_row_of(self, ch: str) -> str:
        top = "qwertyuiop"
        home = "asdfghjkl"
        bottom = "zxcvbnm"
        c = ch.lower()
        if c in top: return "top"
        if c in home: return "home"
        if c in bottom: return "bottom"
        return "other"

    # --- NEW: adjacency builder for QWERTY ---
    def build_qwerty_adjacency(self):
        rows = [
            "`1234567890-=",
            "qwertyuiop[]\\",
            "asdfghjkl;'",
            "zxcvbnm,./"
        ]
        neighbors = defaultdict(set)
        for r in rows:
            for i, ch in enumerate(r):
                for j in (i-1, i+1):
                    if 0 <= j < len(r):
                        neighbors[ch].add(r[j])
                        neighbors[ch.upper()].add(r[j].upper() if r[j].isalpha() else r[j])
                # vertically approximate (rough)
        # add some vertical/diagonal approximations for letters
        approx = {
            'q':'1was', 'w':'2qeas', 'e':'3wrsd', 'r':'4etdf', 't':'5ryfg', 'y':'6tugh', 'u':'7yijh', 'i':'8uokj', 'o':'9ipkl', 'p':'0o;[',
            'a':'qwsz', 's':'qwedxza', 'd':'wersfcx', 'f':'ertdgcv', 'g':'rtyfhvb', 'h':'tyugjbn', 'j':'yuikhmn', 'k':'uiojlm,', 'l':'op;k.',
            'z':'asx', 'x':'zsdc', 'c':'xdfv', 'v':'cfgb', 'b':'vghn', 'n':'bhjm', 'm':'njk,'
        }
        for k,v in approx.items():
            for ch in v:
                neighbors[k].add(ch)
                neighbors[k.upper()].add(ch.upper() if ch.isalpha() else ch)
        return neighbors

    def on_key_release(self, event):
        ch = event.char
        now = time.time()
        keys_to_count = set(string.printable)
        if event.keysym == "BackSpace":
            self.key_hits["<backspace>"] += 1
            self.total_keystrokes += 1
            # keystroke log with no expected comparison
            pos = max(0, len(self.entry.get()))
            expected = self.current_sentence[pos-1] if pos-1 < len(self.current_sentence) and pos>0 else None
            self.keystroke_log.append({"t": now, "typed": "<backspace>", "expected": expected, "correct": False})
            return
        if not ch or ch not in keys_to_count or len(ch) != 1:
            return

        self.key_hits[ch] += 1
        self.total_keystrokes += 1

        pos = len(self.entry.get())
        expected = self.current_sentence[pos - 1] if pos - 1 < len(self.current_sentence) and pos > 0 else None
        if expected is not None:
            if ch == expected:
                self.correct_keystrokes += 1
                self.keystroke_log.append({"t": now, "typed": ch, "expected": expected, "correct": True})
            else:
                self.key_mistakes[ch] += 1
                self.row_mistakes[self.keyboard_row_of(expected)] += 1
                self.keystroke_log.append({"t": now, "typed": ch, "expected": expected, "correct": False})
        else:
            self.keystroke_log.append({"t": now, "typed": ch, "expected": None, "correct": False})

    def next_sentence(self):
        if self.sentence_index >= len(self.sent_list):
            self.sentence_index = 0
        self.current_sentence = self.sent_list[self.sentence_index]
        self.sentence_index += 1

        self.sentence_display.config(state="normal")
        self.sentence_display.delete("1.0", tk.END)
        self.sentence_display.insert(tk.END, self.current_sentence)
        self.sentence_display.config(state="disabled")
        self.entry.delete(0, tk.END)

    # --- helper: detect if w_typed is a simple adjacent transposition of w_target ---
    def is_transposition(self, typed, target):
        if len(typed) != len(target) or typed == target:
            return False
        # find mismatches
        diffs = [i for i,(a,b) in enumerate(zip(typed, target)) if a!=b]
        if len(diffs) != 2:
            return False
        i,j = diffs
        if j == i+1 and typed[i]==target[j] and typed[j]==target[i]:
            return True
        return False

    def submit_sentence(self, event=None):
        typed_full = self.entry.get()
        target_words = self.current_sentence.strip().split()
        typed_words = typed_full.strip().split()

        self.total_words += len(target_words)
        for i, w in enumerate(typed_words):
            if i < len(target_words) and w == target_words[i]:
                self.correct_words += 1

        # --- Record detailed word-level history for AI analysis ---
        word_details = []
        transposition_count = 0
        long_word_errors = 0
        long_word_total = 0
        digraph_problem_counts = Counter()

        for i, tgt in enumerate(target_words):
            typed = typed_words[i] if i < len(typed_words) else ""
            correct = (typed == tgt)
            if len(tgt) >= 7:
                long_word_total += 1
                if not correct:
                    long_word_errors += 1
            # digraph issues (sh/ch): if target contains them and word is wrong
            for dg in ("sh","ch"):
                if dg in tgt.lower() and not correct:
                    digraph_problem_counts[dg] += 1

            if not correct and self.is_transposition(typed, tgt):
                transposition_count += 1

            word_details.append({
                "target": tgt,
                "typed": typed,
                "correct": correct
            })

        self.history.append({
            "target": self.current_sentence,
            "typed": typed_full,
            "duration": max(0.001, time.time() - self.start_time),
            "words": word_details,
            "transpositions": transposition_count,
            "long_word_errors": long_word_errors,
            "long_word_total": long_word_total,
            "digraph_issues": dict(digraph_problem_counts)
        })

        # show per-word correctness in colors
        self.sentence_display.config(state="normal")
        self.sentence_display.delete("1.0", tk.END)
        for i, word in enumerate(target_words):
            tag = "correct" if (i < len(typed_words) and typed_words[i] == word) else "wrong"
            self.sentence_display.insert(tk.END, word + " ", tag)
        self.sentence_display.config(state="disabled")

        self.root.after(700, self.next_sentence)
       
    def submit_sentence(self):
        # your code here
        pass
    def update_timer(self):   # <-- make sure this line has 4 spaces
        elapsed = int(time.time() - self.start_time)
        remaining = self.time_limit - elapsed
        if remaining >= 0:
            self.timer_label.config(text=f"Time Left: {remaining}s")
            self.root.after(1000, self.update_timer)
        else:
            try:
                self.entry.configure(state="disabled")
            except:
                pass

            btns_frame = tk.Frame(self.root, bg=self.COLOR_BG)
            btns_frame.pack(pady=20)
            self.create_styled_button(btns_frame, "Show Results", self.show_results, width=18).pack()
    # ---------- PAGE 4 (Results) ----------
        # ---------- PAGE 4 (Results) ----------
    def show_results(self):
        self.clear_window()
        elapsed_time = max(0.001, time.time() - self.start_time)
        words_typed = self.correct_keystrokes / 5  # standard definition
        wpm = int((words_typed / elapsed_time) * 60)
        accuracy = int((self.correct_keystrokes / self.total_keystrokes) * 100) if self.total_keystrokes > 0 else 0
        cpm = int((self.correct_keystrokes / elapsed_time) * 60)

        # --- Result Page Layout ---
        main_frame = tk.Frame(self.root, bg=self.COLOR_BG)
        main_frame.pack(expand=True)

        # Title
        tk.Label(
            main_frame,
            text="Time's Up! 🥳",
            font=self.FONT_SUBHEADING,
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        ).pack(pady=16)

        # Results section
        results_frame = tk.Frame(main_frame, bg=self.COLOR_SECONDARY, padx=40, pady=20)
        results_frame.pack(pady=20)

        tk.Label(
            results_frame,
            text=f"WPM: {wpm}",
            font=self.FONT_BODY_BOLD,
            bg=self.COLOR_SECONDARY,
            fg=self.COLOR_TEXT
        ).pack(pady=6)

        tk.Label(
            results_frame,
            text=f"CPM: {cpm}",
            font=self.FONT_BODY_BOLD,
            bg=self.COLOR_SECONDARY,
            fg=self.COLOR_TEXT
        ).pack(pady=6)

        tk.Label(
            results_frame,
            text=f"Accuracy: {accuracy}%",
            font=self.FONT_BODY_BOLD,
            bg=self.COLOR_SECONDARY,
            fg=self.COLOR_TEXT
        ).pack(pady=6)

        # Buttons
        self.create_styled_button(
            main_frame,
            "AI Feedback & Analysis",
            self.show_feedback_page,
            width=22
        ).pack(pady=18)

        self.create_styled_button(
            main_frame,
            "Try Again",
            self.show_selection_page,
            width=16
        ).pack(pady=8)

    # ---------- AI ANALYSIS HELPERS ----------

    # adjacent-key error ratio from keystroke_log
    def analyze_adjacent_errors(self):
        adjacent_errors = 0
        total_errors = 0
        for k in self.keystroke_log:
            exp = k["expected"]
            if exp is None: 
                continue
            if not k["correct"]:
                total_errors += 1
                typed = k["typed"]
                # normalize to same case baseline
                e = exp
                t = typed
                # check neighbors (case-sensitive map contains both cases)
                if t in self.adjacent_map.get(e, set()):
                    adjacent_errors += 1
                # also check lower-case mapping if needed
                elif t.lower() in self.adjacent_map.get(e.lower(), set()):
                    adjacent_errors += 1
        return adjacent_errors, total_errors

    def analyze_rhythm(self):
        # compute inter-key intervals
        times = [k["t"] for k in self.keystroke_log if len(k["typed"]) == 1]
        if len(times) < 3:
            return {"overall_avg": None, "before_caps_avg": None, "before_punct_avg": None}

        intervals = []
        for i in range(1, len(self.keystroke_log)):
            prev = self.keystroke_log[i-1]
            curr = self.keystroke_log[i]
            if len(prev["typed"]) == 0 or len(curr["typed"]) == 0:
                continue
            dt = curr["t"] - prev["t"]
            intervals.append((dt, curr))  # interval leading into 'curr' char

        if not intervals:
            return {"overall_avg": None, "before_caps_avg": None, "before_punct_avg": None}

        overall_avg = sum(dt for dt,_ in intervals) / len(intervals)

        # before capital letters (expected uppercase)
        caps_intervals = [dt for dt,k in intervals if (k["expected"] is not None and k["expected"].isalpha() and k["expected"].upper()==k["expected"] and k["expected"].lower()!=k["expected"])]
        caps_avg = (sum(caps_intervals)/len(caps_intervals)) if caps_intervals else None

        # before punctuation
        punct_set = set(".,;:!?\"'()[]{}-—/\\")
        punct_intervals = [dt for dt,k in intervals if (k["expected"] in punct_set)]
        punct_avg = (sum(punct_intervals)/len(punct_intervals)) if punct_intervals else None

        return {"overall_avg": overall_avg, "before_caps_avg": caps_avg, "before_punct_avg": punct_avg}

    def analyze_transpositions_and_cognitive_load(self):
        total_transpositions = 0
        long_err = 0
        long_tot = 0
        digraph_issues = Counter()
        for h in self.history:
            total_transpositions += h.get("transpositions", 0)
            long_err += h.get("long_word_errors", 0)
            long_tot += h.get("long_word_total", 0)
            for k,v in h.get("digraph_issues", {}).items():
                digraph_issues[k] += v
        return total_transpositions, long_err, long_tot, dict(digraph_issues)

    def percent_change(self, base, new):
        if base is None or new is None or base <= 0:
            return None
        return (new - base) / base * 100.0

    def predict_progress(self, milestone_wpm=70, window=10):
        """Simple linear regression y = a + b*x over last `window` sessions where x = 0..n-1, y = wpm."""
        if not self.progress_history:
            return {"can_predict": False}

        data = self.progress_history[-window:]
        n = len(data)
        if n < 3:
            # too few points for reliable prediction
            return {"can_predict": False, "avg_wpm": data[-1]["wpm"] if n else 0}

        ys = [d["wpm"] for d in data]
        xs = list(range(n))
        mean_x = sum(xs)/n
        mean_y = sum(ys)/n
        Sxx = sum((x-mean_x)**2 for x in xs)
        Sxy = sum((x-mean_x)*(y-mean_y) for x,y in zip(xs,ys))
        if Sxx == 0:
            return {"can_predict": False, "avg_wpm": mean_y}

        slope = Sxy / Sxx
        intercept = mean_y - slope*mean_x

        # plateau detection: small slope and small variance in last 5
        plateau = False
        if n >= 5:
            last5 = ys[-5:]
            if statistics.pstdev(last5) <= 2 and abs(slope) < 0.5:
                plateau = True

        # forecast sessions to milestone
        sessions_to_goal = None
        if slope > 0:
            # solve milestone = intercept + slope*x  => x = (milestone - intercept)/slope
            x_goal = (milestone_wpm - intercept) / slope
            if x_goal > xs[-1]:
                sessions_to_goal = max(0, math.ceil(x_goal - xs[-1]))
            else:
                sessions_to_goal = 0
        return {
            "can_predict": True,
            "slope": slope,
            "intercept": intercept,
            "avg_wpm": mean_y,
            "plateau": plateau,
            "sessions_to_goal": sessions_to_goal
        }

    # ---------- PAGE 5 (AI Feedback) ----------
    def show_feedback_page(self):
        self.clear_window()
        
        tk.Label(self.root, text="AI Feedback & Analysis", font=self.FONT_SUBHEADING,
                 bg=self.COLOR_BG, fg=self.COLOR_TEXT).pack(pady=10)

        report = ScrolledText(self.root, font=self.FONT_FEEDBACK, height=24, wrap="word",
                              bg=self.COLOR_WHITE, fg=self.COLOR_TEXT, padx=12, pady=12,
                              relief="flat", borderwidth=2)
        report.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        # --- Base metrics (existing) ---
        elapsed_time = max(0.001, time.time() - self.start_time)
        total_keys = self.total_keystrokes
        correct_keys = self.correct_keystrokes
        mistakes_total = total_keys - correct_keys if total_keys >= correct_keys else 0
        top_mistakes = self.key_mistakes.most_common(5)
        top_hits = self.key_hits.most_common(5)
        row_totals = {
            "top": self.row_mistakes.get("top", 0),
            "home": self.row_mistakes.get("home", 0),
            "bottom": self.row_mistakes.get("bottom", 0),
            "other": self.row_mistakes.get("other", 0),
        }
        worst_row = max(row_totals, key=row_totals.get) if any(row_totals.values()) else "n/a"
        total_words = self.total_words
        correct_words = self.correct_words
        accuracy_words = (correct_words / total_words * 100) if total_words else 0.0
        wpm = (correct_words / elapsed_time) * 60 if elapsed_time > 0 else 0
        cpm = (correct_keys / elapsed_time) * 60 if elapsed_time > 0 else 0

        def line(s=""):
            report.insert(tk.END, s + "\n")

        # ---------- EXISTING REPORT ----------
        line("— SESSION SUMMARY —")
        line(f"Elapsed time:              {elapsed_time:6.1f} s")
        line(f"Keystrokes (total/correct/mistakes): {total_keys}/{correct_keys}/{mistakes_total}")
        line(f"Speed:                     {wpm:5.1f} WPM, {cpm:5.1f} CPM")
        line(f"Word accuracy:             {accuracy_words:5.1f}%")
        line()
        line("— MOST FREQUENT MISTAKE KEYS —")
        if top_mistakes:
            for k, v in top_mistakes: line(f"  '{k}': {v} times")
        else: line("  No frequent mistake keys detected. Great control! 🎉")
        line()
        line("— MOST PRESSED KEYS —")
        if top_hits:
            for k, v in top_hits: line(f"  '{k}': {v} hits")
        else: line("  No key hits recorded yet.")
        line()
        line("— ROW WEAKNESS (expected-char row for mistakes) —")
        line(f"  Top row mistakes:    {row_totals['top']}")
        line(f"  Home (middle) row:   {row_totals['home']}")
        line(f"  Bottom row mistakes: {row_totals['bottom']}")
        line(f"  Other characters:    {row_totals['other']}")
        line(f"  Row needing practice: {worst_row.upper() if worst_row!='n/a' else 'n/a'}")
        line()
        line("— STRENGTHS —")
        if accuracy_words >= 95: line("  • Excellent accuracy — keep pushing speed.")
        elif wpm >= 50: line("  • Solid speed — refine accuracy for fewer corrections.")
        else: line("  • Building a good base — steady practice will raise both speed and accuracy.")
        if mistakes_total == 0: line("  • Zero mistakes recorded across keystrokes. 🎯")
        line()

        # ---------- NEW: ADVANCED ERROR PATTERNS ----------
        line("— ADVANCED ERROR PATTERN RECOGNITION 🧠 —")

        # Transpositions & Cognitive load
        transpositions, long_err, long_tot, digraph_issues = self.analyze_transpositions_and_cognitive_load()
        if transpositions:
            line(f"  • Transposition errors detected: {transpositions} (e.g., 'teh' instead of 'the').")
        else:
            line("  • No transposition errors detected.")

        # Adjacent-key errors
        adj_err, total_err = self.analyze_adjacent_errors()
        if total_err > 0:
            pct_adj = (adj_err/total_err*100.0)
            line(f"  • Adjacent-key error indicator: {adj_err}/{total_err} mistakes (~{pct_adj:.1f}%). This points to finger accuracy issues.")
        else:
            line("  • Not enough error data for adjacent-key analysis yet.")

        # Rhythm and pacing
        rhythm = self.analyze_rhythm()
        overall = rhythm.get("overall_avg")
        caps_avg = rhythm.get("before_caps_avg")
        punct_avg = rhythm.get("before_punct_avg")
        slowdown_caps = self.percent_change(overall, caps_avg)
        slowdown_punct = self.percent_change(overall, punct_avg)

        if overall is not None:
            line(f"  • Average inter-key interval: {overall*1000:.0f} ms.")
            if slowdown_caps is not None:
                if slowdown_caps > 25:
                    line(f"  • You slow down by ~{slowdown_caps:.0f}% before CAPITAL letters. Practice sentences with more capitalization.")
                else:
                    line("  • No significant slowdown before capital letters.")
            else:
                line("  • Not enough capital-letter data to assess slowdown.")
            if slowdown_punct is not None:
                if slowdown_punct > 25:
                    line(f"  • You slow down by ~{slowdown_punct:.0f}% before punctuation. Try rhythm drills that include , . ! ?")
                else:
                    line("  • No significant slowdown before punctuation.")
            else:
                line("  • Not enough punctuation data to assess slowdown.")
        else:
            line("  • Not enough keystroke timing data for rhythm analysis.")
        
        # Cognitive load: long words
        if long_tot > 0:
            long_err_rate = long_err / long_tot * 100.0
            line(f"  • Long-word difficulty: {long_err}/{long_tot} long words incorrect (~{long_err_rate:.1f}%). This suggests cognitive load.")
        else:
            line("  • No long words encountered to assess cognitive load.")

        if digraph_issues:
            ss = ", ".join([f"{k}: {v}" for k,v in digraph_issues.items()])
            line(f"  • Digraph challenges detected → {ss}.")
        else:
            line("  • No specific digraph (sh/ch) challenges detected.")
        line()

        # ---------- NEW: PERSONALIZED PRACTICE GENERATION ----------
        line("— PERSONALIZED PRACTICE 🎯 —")
        custom_drill = self.generate_custom_practice(
            worst_row=worst_row,
            digraph_issues=digraph_issues,
            slowdown_caps=(slowdown_caps if slowdown_caps is not None and slowdown_caps>25 else 0),
            slowdown_punct=(slowdown_punct if slowdown_punct is not None and slowdown_punct>25 else 0),
            adjacent_problem=(adj_err/total_err>0.3 if total_err>0 else False)
        )
        line(custom_drill)
        line()

        # ---------- NEW: PREDICTIVE PERFORMANCE ANALYSIS ----------
        line("— PREDICTIVE PERFORMANCE 📊 —")
        pred = self.predict_progress(milestone_wpm=70, window=10)
        if pred.get("can_predict"):
            line(f"  • Recent average WPM: {pred['avg_wpm']:.1f}")
            if pred["plateau"]:
                line("  • Progress plateau detected in recent sessions. Suggest shifting to accuracy-focused drills for a while.")
            if pred.get("sessions_to_goal") is not None:
                if pred["sessions_to_goal"] == 0:
                    line("  • You're at or above the 70 WPM milestone. Great job!")
                else:
                    line(f"  • At your current rate, you’re on track to hit 70 WPM in ~{pred['sessions_to_goal']} more sessions.")
            else:
                line("  • Unable to confidently forecast sessions to milestone from current trend.")
        else:
            if self.progress_history:
                last = self.progress_history[-1]["wpm"]
                line(f"  • Not enough session data for trend analysis yet. Last WPM: {last}. Keep logging sessions!")
            else:
                line("  • No past sessions recorded yet. Complete more tests to enable forecasts.")
        line()

        # ---------- EXISTING SUGGESTIONS ----------
        line("— SUGGESTED PRACTICE PLAN —")
        if worst_row != "n/a": line(f"  • Spend 5–10 minutes on {worst_row.upper()} row drills.")
        if top_mistakes:
            hardest = ", ".join([f"'{k}'" for k, _ in top_mistakes[:3]])
            line(f"  • Focus on these keys: {hardest}. Try slow, rhythmic repeats (e.g., {hardest}…)")
        line("  • Practice short bursts with focus on posture and consistent finger placement.")
        line("  • Aim for smoothness first; speed will follow.")
        line()
        line("— SENTENCE HISTORY —")
        if self.history:
            for i, h in enumerate(self.history, 1):
                t_words = h["target"].split()
                y_words = h["typed"].split()
                cw = sum(1 for idx, w in enumerate(y_words) if idx < len(t_words) and w == t_words[idx])
                line(f"  {i:2d}. Words correct: {cw}/{len(t_words)}")
        else: line("  No sentences submitted yet.")
        
        report.configure(state="disabled")

        btns_frame = tk.Frame(self.root, bg=self.COLOR_BG)
        btns_frame.pack(pady=10)
        self.create_styled_button(btns_frame, "Back to Selection", self.show_selection_page, font=self.FONT_SMALL, width=16).pack()

    # --- NEW: custom practice generator based on weaknesses ---
    def generate_custom_practice(self, worst_row, digraph_issues, slowdown_caps, slowdown_punct, adjacent_problem):
        lines = []

        # Row-focused drills
        if worst_row == "bottom":
            lines.append("Zany zebras buzzed by; mix, nix, and maximize nimble moves.")
        elif worst_row == "top":
            lines.append("Quick wizards type quirky queries to prove quality.")
        elif worst_row == "home":
            lines.append("Dash and splash as Della juggles joyful, agile tasks.")
        else:
            lines.append("Practice smooth, steady sentences to solidify rhythm and control.")

        # Digraphs
        if digraph_issues.get("sh", 0) > 0:
            lines.append("She saw shimmering shells shine as shadows shifted on the shore.")
        if digraph_issues.get("ch", 0) > 0:
            lines.append("Charlie chose a cheerful chocolate cheesecake for the chilly church picnic.")

        # Capital slowdown
        if slowdown_caps and slowdown_caps > 0:
            lines.append("Alice And Bob Bring Big Boxes; Carol Carefully Counts Cards.")
        
        # Punctuation slowdown
        if slowdown_punct and slowdown_punct > 0:
            lines.append("Wait, what? Stop! Listen: practice pauses—then proceed.")
        
        # Adjacent key issues
        if adjacent_problem:
            lines.append("Refine finger accuracy: ten tiny taps; ever even edges; near keys, never nudge.")

        # Default filler to reach a paragraph feel
        if len(lines) < 3:
            lines.append("Keep a calm cadence; precision first, then pace. Focus, breathe, and flow.")

        return " ".join(lines)

    # ---------- Helpers ----------
    def clear_window(self):
        for w in self.root.winfo_children():
            w.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TypingCoachApp(root)
    root.mainloop()

