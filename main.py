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

    def abstractive_summary(self, extractive_summary, max_length):
        abstractive_summary = self.abstractive_summarizer(
            extractive_summary,
            max_length=max_length,
            min_length=max_length - 20,
            do_sample=True,
            temperature=1.2,
            top_k=50,
            top_p=0.95
        )[0]['summary_text']
        return abstractive_summary

class SummaryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Summarizer")
        self.root.geometry("700x700")
        self.root.configure(bg="#f0f0f0")  # Light gray background

        # Custom font
        self.title_font = tkFont.Font(family='Helvetica', size=16, weight='bold')
        self.label_font = tkFont.Font(family='Helvetica', size=12)

        # Summarizer object
        self.summarizer = Summarizer()


        title_label = tk.Label(root, text="Text Summarizer", font=self.title_font, bg="#f0f0f0")
        title_label.pack(pady=10)

        tk.Label(root, text="Input Text:", font=self.label_font, bg="#f0f0f0").pack()
        self.input_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10)
        self.input_text.pack(pady=5)


        tk.Label(root, text="Extractive Summary Lines:", font=self.label_font, bg="#f0f0f0").pack()
        self.extractive_lines = tk.Entry(root, width=10)
        self.extractive_lines.pack(pady=5)
        self.extractive_lines.insert(0, "3")  # Default to 3 lines

        tk.Label(root, text="Abstractive Summary Lines (approx.):", font=self.label_font, bg="#f0f0f0").pack()
        self.abstractive_lines = tk.Entry(root, width=10)
        self.abstractive_lines.pack(pady=5)
        self.abstractive_lines.insert(0, "3")  # Default to 3 lines

        self.summarize_button = tk.Button(root, text="Generate Summaries", command=self.generate_summaries, bg="#4CAF50", fg="white", font=self.label_font)
        self.summarize_button.pack(pady=10)

        tk.Label(root, text="Extractive Summary:", font=self.label_font, bg="#f0f0f0").pack()
        self.extractive_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=5, state='disabled')
        self.extractive_text.pack(pady=5)

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
            # Get user-specified line counts
            extractive_lines = int(self.extractive_lines.get())
            abstractive_lines = int(self.abstractive_lines.get())
            max_length_abstractive = abstractive_lines * 20  # Approximate max length per line

            # Generate extractive summary
            extractive = self.summarizer.extractive_summary(input_data, num_sentences=extractive_lines)
            self.extractive_text.insert(tk.END, extractive)

            # Generate abstractive summary
            if len(extractive.split()) > 5:  # Ensure sufficient length
                abstractive = self.summarizer.abstractive_summary(extractive, max_length=max_length_abstractive)

                # Check similarity and output
                if abstractive.strip() == extractive.strip():
                    abstractive = "The abstractive summary is too similar to the extractive summary."
                self.abstractive_text.insert(tk.END, abstractive)
            else:
                self.abstractive_text.insert(tk.END, "Input text is too short for meaningful abstractive summarization.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

        # Disable editing for output fields
        self.extractive_text.config(state='disabled')
        self.abstractive_text.config(state='disabled')

    def show_help(self):
        help_message = ("To use this summarization tool:\n\n"
                        "1. Enter the text you want to summarize in the 'Input Text' area.\n"
                        "2. Specify the number of lines for extractive and abstractive summaries.\n"
                        "3. Click on 'Generate Summaries'.\n"
                        "4. The summaries will appear below, which you can copy.")
        messagebox.showinfo("Help", help_message)

root = tk.Tk()
app = SummaryApp(root)
root.mainloop()
