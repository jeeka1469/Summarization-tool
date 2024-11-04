import tkinter as tk
from tkinter import scrolledtext, messagebox
from tkinter import font as tkFont
from transformers import pipeline
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK data is available
nltk.download('punkt')

class Summarizer:
    def __init__(self):
        self.extractor = TfidfVectorizer()
        self.abstractive_summarizer = pipeline("summarization", model="t5-small")

    def extractive_summary(self, text, num_sentences=3):
        sentences = nltk.sent_tokenize(text)
        tfidf_matrix = self.extractor.fit_transform(sentences)
        cosine_matrix = cosine_similarity(tfidf_matrix)
        summary_indices = cosine_matrix.sum(axis=1).argsort()[-num_sentences:]
        extractive_summary = " ".join([sentences[i] for i in sorted(summary_indices)])
        return extractive_summary

    def abstractive_summary(self, extractive_summary):
        abstractive_summary = self.abstractive_summarizer(extractive_summary, max_length=60, min_length=40, do_sample=False)[0]['summary_text']
        return abstractive_summary

class SummaryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Summarizer")
        self.root.geometry("700x600")
        self.root.configure(bg="#f0f0f0")  # Light gray background

        # Custom font
        self.title_font = tkFont.Font(family='Helvetica', size=16, weight='bold')
        self.label_font = tkFont.Font(family='Helvetica', size=12)

        # Summarizer object
        self.summarizer = Summarizer()

        # Title Label
        title_label = tk.Label(root, text="Text Summarizer", font=self.title_font, bg="#f0f0f0")
        title_label.pack(pady=10)

        # Input text field
        tk.Label(root, text="Input Text:", font=self.label_font, bg="#f0f0f0").pack()
        self.input_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10)
        self.input_text.pack(pady=5)

        # Button to generate summary
        self.summarize_button = tk.Button(root, text="Generate Summaries", command=self.generate_summaries, bg="#4CAF50", fg="white", font=self.label_font)
        self.summarize_button.pack(pady=10)

        # Extractive summary output
        tk.Label(root, text="Extractive Summary:", font=self.label_font, bg="#f0f0f0").pack()
        self.extractive_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=5, state='disabled')
        self.extractive_text.pack(pady=5)

        # Abstractive summary output
        tk.Label(root, text="Abstractive Summary:", font=self.label_font, bg="#f0f0f0").pack()
        self.abstractive_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=5, state='disabled')
        self.abstractive_text.pack(pady=5)

        # Help button
        self.help_button = tk.Button(root, text="Help", command=self.show_help, bg="#2196F3", fg="white", font=self.label_font)
        self.help_button.pack(pady=10)

    def generate_summaries(self):
        # Clear previous summaries
        self.extractive_text.config(state='normal')
        self.extractive_text.delete(1.0, tk.END)
        self.abstractive_text.config(state='normal')
        self.abstractive_text.delete(1.0, tk.END)

        # Get the input text
        input_data = self.input_text.get(1.0, tk.END).strip()
        if not input_data:
            messagebox.showwarning("Warning", "Please enter text to summarize.")
            return

        try:
            # Generate extractive summary
            extractive = self.summarizer.extractive_summary(input_data)
            self.extractive_text.insert(tk.END, extractive)

            # Generate abstractive summary
            abstractive = self.summarizer.abstractive_summary(extractive)
            self.abstractive_text.insert(tk.END, abstractive)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

        # Disable editing for output fields
        self.extractive_text.config(state='disabled')
        self.abstractive_text.config(state='disabled')

    def show_help(self):
        help_message = ("To use this summarization tool:\n\n"
                        "1. Enter the text you want to summarize in the 'Input Text' area.\n"
                        "2. Click on 'Generate Summaries'.\n"
                        "3. The extractive and abstractive summaries will be displayed below.\n"
                        "4. You can copy the summaries from the output areas.")
        messagebox.showinfo("Help", help_message)

# Run the application
root = tk.Tk()
app = SummaryApp(root)
root.mainloop()