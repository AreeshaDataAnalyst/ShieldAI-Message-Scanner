# ShieldAI-Message-Scanner
Detect suspicious messages, understand the warning signs, and make safer decisions before you click.

# ShieldAI - AI-Powered Scam and Spam Detection

ShieldAI is an educational Streamlit application that screens SMS,
email excerpts, and chat messages for spam-like patterns.

## Features

- Spam vs ham classification using TF-IDF and machine learning
- Logistic Regression and Multinomial Naive Bayes comparison
- Stratified 80/20 train-test split
- Suspicious phrase and URL pattern detection
- Heuristic risk level and model confidence
- Optional Groq-generated explanations and safe replies
- PyTorch BiLSTM training and test evaluation
- Confusion matrix and model metrics dashboard

## Project structure

```text
ShieldAI/
├── app.py
├── train_models.py
├── train_bilstm.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── data/
│   └── SpamSmsDataset.xlsx
└── models/
    ├── spam_model.joblib
    └── metrics.json
```

Model files are generated after training.

## Setup

Python 3.10 or 3.11 is recommended.

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install packages:

```bash
pip install -r requirements.txt
```

## Dataset

Place `SpamSmsDataset.xlsx` in the `data` directory.

The loader supports Excel workbooks and CSV-formatted files. Required
columns are `message` and `label`. Labels must be `spam` and `ham`.

## Train and evaluate machine learning

```bash
python train_models.py
```

This compares Logistic Regression and Multinomial Naive Bayes using a
stratified 80/20 train-test split. It selects the model with the higher
spam F1-score and saves the model and metrics in `models/`.

## Train the BiLSTM model

```bash
python train_bilstm.py
```

This trains a separate PyTorch BiLSTM and saves its test metrics.

## Configure Groq

Copy `.env.example` to `.env` and add your API key:

```env
GROQ_API_KEY=your_actual_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Keep `.env` private. Never commit API keys to GitHub.

## Run the application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal.

## GitHub

```bash
git init
git add .
git commit -m "Build ShieldAI scam spam detector"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your repository URL.

## Limitations

This is an educational classifier, not a guarantee of fraud detection.
The dataset represents its own spam/ham distribution and may not reflect
modern phishing attacks. A ham prediction does not prove that a message
is safe. Verify sensitive requests independently.

When Groq is enabled, submitted message content is sent to the Groq API
for analysis. Do not submit confidential messages.
