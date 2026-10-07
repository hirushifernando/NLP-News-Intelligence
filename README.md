# 📰 AI-Based News Article Topic Classification & Intelligent Article Summarization

A deep learning-based Natural Language Processing (NLP) system that automatically **classifies news articles into four topics** and generates **concise summaries** using a pretrained Transformer model.

The project implements and compares two deep learning approaches for news classification:

* **PyTorch** — BiLSTM with Attention
* **TensorFlow/Keras** — Bidirectional LSTM

For intelligent summarization, the project uses **Hugging Face Transformers with T5-small** and evaluates generated summaries using **ROUGE metrics**.

A **Streamlit web application** combines both capabilities into an interactive News Intelligence system.

---

## 🎯 Project Objectives

The main objectives of this project are to:

* Classify news articles into predefined topics.
* Build a deep learning NLP classifier using PyTorch.
* Implement a second classifier using TensorFlow/Keras.
* Compare the performance of PyTorch and Keras implementations.
* Use Hugging Face datasets and Transformers.
* Generate automatic summaries of news articles.
* Evaluate summarization quality using ROUGE.
* Develop an interactive Streamlit application.
* Demonstrate an end-to-end NLP pipeline from preprocessing to deployment.

---

## 🧠 System Overview

The system contains two main NLP tasks:

### 1. News Topic Classification

```text
News Article
     ↓
Text Cleaning
     ↓
Tokenization
     ↓
Vocabulary Mapping
     ↓
Padding / Truncation
     ↓
BiLSTM Model
     ↓
Attention Mechanism
     ↓
News Topic
```

### 2. News Article Summarization

```text
News Article
     ↓
T5 Tokenizer
     ↓
T5-small Transformer
     ↓
Beam Search Generation
     ↓
Generated Summary
     ↓
ROUGE Evaluation
```

---

## 🗂️ News Categories

The classification model predicts one of four categories:

| Label | Category    |
| ----: | ----------- |
|     0 | 🌍 World    |
|     1 | ⚽ Sports    |
|     2 | 💼 Business |
|     3 | 🔬 Sci/Tech |

---

## 📊 Dataset

### News Classification Dataset

The classification task uses the **AG News dataset** accessed through Hugging Face:

**Dataset:** `SetFit/ag_news`

Dataset distribution:

| Split      | Samples |
| ---------- | ------: |
| Training   | 120,000 |
| Validation |  12,000 |
| Testing    |   7,600 |

The original training set was further divided into:

* Training: **108,000**
* Validation: **12,000**
* Test: **7,600**

The dataset contains four balanced news categories with approximately **30,000 training examples per category**.

### Summarization Dataset

For summarization evaluation, the project uses the **CNN/DailyMail** dataset:

`phoenix60064/cnn_dailymail`

| Split      | Samples |
| ---------- | ------: |
| Training   | 287,113 |
| Validation |  13,368 |
| Testing    |  11,490 |

Unlike AG News, CNN/DailyMail provides human-written reference summaries, making it suitable for ROUGE-based evaluation.

---

## 🧹 Text Preprocessing

The news classification pipeline applies the following preprocessing steps:

1. Convert text to lowercase.
2. Remove URLs.
3. Remove HTML tags.
4. Remove non-alphabetic characters.
5. Normalize whitespace.
6. Split text into tokens.
7. Convert words into vocabulary indices.
8. Replace unknown words with `<unk>`.
9. Pad or truncate sequences to a fixed length.

### Vocabulary

* Vocabulary size: **20,000**
* `<pad>` token: `0`
* `<unk>` token: `1`
* Maximum sequence length: **100**

---

# 🧠 PyTorch Classification Model

The main classification model was implemented using PyTorch.

### Architecture

```text
Input Text
    ↓
Embedding
    ↓
Bidirectional LSTM
    ↓
Attention Mechanism
    ↓
Dropout
    ↓
Fully Connected Layer
    ↓
Softmax
    ↓
4 News Categories
```

### Model Configuration

| Parameter               |         Value |
| ----------------------- | ------------: |
| Vocabulary Size         |        20,000 |
| Embedding Dimension     |           128 |
| LSTM Hidden Dimension   |           128 |
| LSTM Direction          | Bidirectional |
| Dropout                 |           0.3 |
| Classes                 |             4 |
| Optimizer               |          Adam |
| Learning Rate           |         0.001 |
| Loss Function           | Cross Entropy |
| Batch Size              |            64 |
| Epochs                  |             5 |
| Maximum Sequence Length |           100 |

The best validation checkpoint was obtained at **Epoch 2**.

Although training continued to Epoch 5, validation performance started to decrease after Epoch 2 while training accuracy continued to increase, indicating overfitting.

---

# 🔄 TensorFlow/Keras Classification Model

A second implementation was developed using TensorFlow/Keras to compare the classification performance.

### Architecture

```text
Input Sequence
     ↓
Embedding
     ↓
Bidirectional LSTM
     ↓
Dropout
     ↓
Dense + Softmax
     ↓
4 News Categories
```

### Configuration

| Parameter           |                           Value |
| ------------------- | ------------------------------: |
| Vocabulary Size     |                          20,000 |
| Embedding Dimension |                             128 |
| LSTM Units          |                             128 |
| Dropout             |                             0.3 |
| Classes             |                               4 |
| Optimizer           |                            Adam |
| Learning Rate       |                           0.001 |
| Loss                | Sparse Categorical Crossentropy |
| Batch Size          |                              64 |
| Epochs              |                               5 |

The best validation performance was obtained at **Epoch 3**.

---

# 📈 Classification Results

## PyTorch Results

The PyTorch model achieved:

**Test Accuracy: 91.18%**

| Category       | Precision |   Recall | F1-Score |
| -------------- | --------: | -------: | -------: |
| World          |      0.93 |     0.90 |     0.92 |
| Sports         |      0.95 |     0.98 |     0.96 |
| Business       |      0.89 |     0.86 |     0.88 |
| Sci/Tech       |      0.88 |     0.91 |     0.89 |
| **Macro Avg.** |  **0.91** | **0.91** | **0.91** |

### PyTorch Confusion Matrix

|              | World | Sports | Business | Sci/Tech |
| ------------ | ----: | -----: | -------: | -------: |
| **World**    |  1707 |     60 |       67 |       66 |
| **Sports**   |    21 |   1853 |       15 |       11 |
| **Business** |    62 |     29 |     1642 |      167 |
| **Sci/Tech** |    39 |     15 |      118 |     1728 |

The main classification challenge was distinguishing **Business** and **Sci/Tech** articles.

---

## Keras Results

The Keras model achieved:

**Test Accuracy: 91.38%**

| Category       | Precision |   Recall | F1-Score |
| -------------- | --------: | -------: | -------: |
| World          |      0.92 |     0.91 |     0.92 |
| Sports         |      0.96 |     0.98 |     0.97 |
| Business       |      0.88 |     0.88 |     0.88 |
| Sci/Tech       |      0.89 |     0.89 |     0.89 |
| **Macro Avg.** |  **0.91** | **0.91** | **0.91** |

### Keras Confusion Matrix

|              | World | Sports | Business | Sci/Tech |
| ------------ | ----: | -----: | -------: | -------: |
| **World**    |  1731 |     45 |       72 |       52 |
| **Sports**   |    23 |   1853 |       16 |        8 |
| **Business** |    68 |     16 |     1675 |      141 |
| **Sci/Tech** |    60 |     19 |      135 |     1686 |

---

# ⚖️ PyTorch vs Keras

| Metric          |    PyTorch |      Keras |
| --------------- | ---------: | ---------: |
| Test Accuracy   | **91.18%** | **91.38%** |
| Test Loss       |     0.2667 | **0.2631** |
| Macro Precision |       0.91 |       0.91 |
| Macro Recall    |       0.91 |       0.91 |
| Macro F1        |       0.91 |       0.91 |

### Comparison

The two implementations achieved very similar performance.

The Keras model achieved a slightly higher test accuracy:

**91.38% vs 91.18%**

The difference is small, showing that both implementations provide comparable classification performance for this dataset.

---

# 🤗 Hugging Face & T5 Summarization

For automatic news summarization, the project uses:

**Model:** `t5-small`

The model is loaded using the Hugging Face Transformers library.

### Summarization Pipeline

```text
Article
   ↓
"T5 summarize:" Prefix
   ↓
T5 Tokenizer
   ↓
T5-small
   ↓
Beam Search
   ↓
Generated Summary
```

The summarizer uses:

* Maximum input length: 512 tokens
* Maximum summary length: 100 tokens
* Minimum summary length: 20 tokens
* Beam search: 4 beams

Example:

**Input:**

```text
A long news article containing multiple paragraphs of information...
```

**Output:**

```text
A shorter automatically generated summary containing the main information.
```

---

# 📏 Summarization Evaluation

The generated summaries were evaluated using **ROUGE**.

The evaluation was performed on **100 test articles** from the CNN/DailyMail dataset.

| Metric  | Average F1 Score |
| ------- | ---------------: |
| ROUGE-1 |       **0.3324** |
| ROUGE-2 |       **0.1361** |
| ROUGE-L |       **0.2478** |

### Interpretation

* **ROUGE-1** measures unigram overlap between the generated and reference summaries.
* **ROUGE-2** measures bigram overlap.
* **ROUGE-L** measures the longest common subsequence.

The results provide a quantitative evaluation of the generated summaries against human-written reference summaries.

---

# 🌐 Streamlit Application

The project includes an interactive Streamlit web application called:

**News Intelligence AI**

The application accepts a news article and provides:

### 🔎 Topic Classification

* Predicted news category
* Prediction confidence

### 📝 Intelligent Summarization

* Automatically generated article summary

### Application Flow

```text
User enters news article
          ↓
       Analyze
          ↓
   ┌──────┴──────┐
   ↓             ↓
Classification  Summarization
   ↓             ↓
News Topic      T5 Summary
   ↓             ↓
Confidence      Generated Text
```

---

# 📁 Project Structure

```text
NLP-News-Intelligence/
│
├── app/
│   └── app.py
│
├── models/
│   ├── pytorch/
│   │   └── best_news_classifier.pt
│   │
│   └── keras/
│       └── best_news_classifier.keras
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_tokenization.ipynb
│   ├── 05_keras_model.ipynb
│   ├── 06_model_comparison.ipynb
│   └── 07_summarization.ipynb
│
├── results/
│   ├── metrics/
│   ├── plots/
│   └── predictions/
│
├── src/
│   ├── preprocessing/
│   │   ├── cleaning.py
│   │   └── tokenization.py
│   │
│   ├── pytorch/
│   │   └── model.py
│   │
│   ├── summarization/
│   │   └── summarizer.py
│   │
│   └── utils/
│       └── helpers.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/hirushifernando/NLP-News-Intelligence.git
cd NLP-News-Intelligence
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Run the Streamlit Application

From the project root:

```bash
streamlit run app/app.py
```

The application will open locally in your browser.

Default address:

```text
http://localhost:8501
```

---

# 🧪 Model Testing

The saved models were independently tested after training to verify that they produce the same evaluation results as the original training notebooks.

### Verified Results

| Model   | Verified Test Accuracy |
| ------- | ---------------------: |
| PyTorch |             **91.18%** |
| Keras   |             **91.38%** |

The verification results are stored in:

```text
results/metrics/
```

---

# 📊 Results and Visualizations

The repository contains:

* Classification reports
* Confusion matrices
* Training accuracy plots
* Training loss plots
* PyTorch/Keras comparison
* ROUGE evaluation results
* Test predictions

These can be found under:

```text
results/
```

---

# 🛠️ Technologies Used

### Programming

* Python

### Deep Learning

* PyTorch
* TensorFlow
* Keras

### NLP

* Hugging Face Transformers
* Hugging Face Datasets
* T5-small
* Tokenization

### Data Processing

* NumPy
* Pandas
* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

### Evaluation

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROUGE-1
* ROUGE-2
* ROUGE-L

### Application

* Streamlit

### Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

# 💡 Key Learning Outcomes

Through this project, the following concepts were explored:

* NLP text preprocessing
* Vocabulary construction
* Sequence tokenization
* Padding and truncation
* Word embeddings
* Bidirectional LSTMs
* Attention mechanisms
* Deep learning model training
* PyTorch model development
* TensorFlow/Keras model development
* Model performance comparison
* Transformer-based NLP
* Text summarization
* ROUGE evaluation
* Model persistence and verification
* Streamlit deployment
* End-to-end ML project organization

---

# 🔮 Future Improvements

Possible future improvements include:

* Fine-tuning a Transformer model specifically for news classification.
* Replacing the word-level BiLSTM classifier with BERT/RoBERTa.
* Improving summarization through fine-tuning on domain-specific news data.
* Adding extractive and abstractive summarization comparison.
* Adding multilingual news classification and summarization.
* Adding explainability for classification predictions.
* Deploying the Streamlit application to a cloud platform.
* Adding automated model testing and CI/CD.
* Supporting batch news article processing.

---

# 📌 Project Status

**Completed**

The project currently includes:

* ✅ Dataset exploration
* ✅ Text preprocessing
* ✅ Tokenization
* ✅ PyTorch classification
* ✅ Keras classification
* ✅ PyTorch vs Keras comparison
* ✅ T5-small summarization
* ✅ ROUGE evaluation
* ✅ Saved model verification
* ✅ Streamlit application
* ✅ Results and visualizations
* ✅ GitHub repository

---

## 👩‍💻 Author

**Hirushi Fernando**

Computer Science Graduate | AI & Machine Learning Enthusiast | Researcher

---

## 📄 License

This project is intended for educational, research, and portfolio purposes.
