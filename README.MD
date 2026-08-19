# 🧠 AutoGrade AI


### LLM-Driven Automatic Answer Assessment & Intelligent Feedback


[![Live Demo](https://img.shields.io/badge/Live%20Demo-AutoGrade%20AI-6366F1?style=for-the-badge)](https://autograde-ai.streamlit.app/)


AutoGrade AI is an intelligent automatic answer assessment system that evaluates student responses using Natural Language Processing, transformer-based semantic analysis, syntactic features, and machine-learning-based scoring.


The system evaluates a student's answer against a given question and produces:


- Overall score out of 10
- Content score out of 5
- Language score out of 2
- Organisation score out of 3
- Answer-specific evaluation explanation
- Personalized improvement feedback


---


## 🚀 Live Demo


### Try AutoGrade AI


👉 https://autograde-ai.streamlit.app/


The deployed application provides a complete interactive assessment workflow:


**Question → Student Answer → NLP Analysis → Score Prediction → Rubric Evaluation → Explanation & Feedback**


---


# 📌 Project Overview


Traditional evaluation of descriptive and essay-type answers is time-consuming and can vary between evaluators.


AutoGrade AI aims to provide a consistent automated assessment process by combining:


- NLP preprocessing
- Transformer-based semantic analysis
- Syntactic feature extraction
- Feature engineering
- Machine-learning-based scoring
- Rubric-based evaluation
- Answer-specific explanation
- Personalized feedback


The system is designed for educational assessment and academic project demonstration purposes.


---


# 🎯 Problem Statement


Manual evaluation of descriptive and essay-type answers requires significant time and effort.


It can also introduce:


- Variation between evaluators
- Delayed feedback
- Difficulty in maintaining consistent assessment criteria
- Limited scalability for large numbers of student responses


AutoGrade AI addresses these challenges through an automated assessment pipeline that combines semantic, syntactic, and machine-learning-based features.


---


# 💡 Proposed Solution


The system accepts two inputs:


1. Essay / Question
2. Student Answer


The answer is processed through multiple assessment stages.


The system then predicts an overall score and derives rubric-wise scores for:


| Criterion | Maximum Score |
🔬 Methodology
1. NLP Preprocessing

The student's answer is normalized using NLP preprocessing techniques.

The preprocessing module performs operations such as:

Tokenization
Stopword removal
Lemmatization
Text normalization

The preprocessing stage prepares the answer for subsequent semantic and syntactic analysis.

2. Semantic Analysis

AutoGrade AI uses the transformer-based Sentence Transformer model:

sentence-transformers/all-MiniLM-L6-v2

MiniLM generates contextual embeddings for the question and student's answer.

The system calculates Question–Answer semantic similarity using cosine similarity.

This provides a measure of how closely the student's response relates to the given question.

The semantic representation is also included in the feature vector used for score prediction.

3. Syntactic Analysis

The system extracts linguistic and readability features from the student's answer.

The current feature set includes:

Average sentence length
Noun ratio
Verb ratio
Adjective ratio
Readability

These features provide additional information about the structure and language quality of the response.

4. Feature Engineering

Semantic and syntactic information are combined into a single feature vector.

The feature set includes:

avg_sentence_length
noun_ratio
verb_ratio
adj_ratio
readability
semantic_similarity
MiniLM answer embedding

The combined feature vector is passed through the trained scaler before score prediction.

5. Score Prediction

The project uses:

Ridge Regression

The trained Ridge regression model is stored as:

models/ridge_model.pkl

The corresponding feature scaler is stored as:

models/scaler.pkl

The trained model produces a prediction that is incorporated into the application's final scoring pipeline.

The final assessment is presented on a 10-point scale.

📊 Assessment Rubric

AutoGrade AI uses a 10-mark rubric:

Content — 5 Marks

Evaluates:

Relevance to the question
Coverage of important concepts
Presence of expected concepts
Adequacy of explanation
Supporting examples where appropriate
Language — 2 Marks

Evaluates:

Grammar
Sentence clarity
Vocabulary
Readability
Overall language quality
Organisation — 3 Marks

Evaluates:

Logical flow
Structure
Ordering of ideas
Coherence
Paragraph organisation
Overall Score
Content       → 5 marks
Language      → 2 marks
Organisation  → 3 marks
────────────────────────
Total         → 10 marks
🤖 Explanation & Feedback

The system does not simply return a numerical score.

It also provides an explanation of the assessment and personalized improvement feedback.

The explanation/feedback module considers the submitted question, answer, semantic similarity, and rubric scores.

The intended output is answer-specific rather than a fixed generic response.

For example, instead of returning only:

Improve grammar and add more details.

the system can provide feedback related to the actual response and its relationship to the question.

The explanation should help answer:

Why did the system give this score?

The feedback can identify strengths and areas where the student's response can be improved.

✨ Key Features
📝 Interactive Assessment

Enter a question and student answer directly through the Streamlit interface.

🧠 Semantic Understanding

Uses MiniLM transformer embeddings to measure semantic similarity.

📐 Syntactic Analysis

Extracts linguistic and readability features from the response.

📊 Machine Learning Scoring

Uses a trained Ridge Regression model as part of the scoring pipeline.

📚 Rubric-Based Evaluation

Provides separate scores for:

Content
Language
Organisation
💬 AI Evaluation Explanation

Provides reasoning related to the assessment.

💡 Personalized Feedback

Provides suggestions based on the submitted response.

⚠️ Relevance Detection

The application can identify responses that may not sufficiently address the given question.

🌐 Cloud Deployment

The application is deployed using Streamlit Community Cloud.

🛠️ Technology Stack
Category	Technology
Programming Language	Python
Application Framework	Streamlit
NLP	NLTK
Linguistic Analysis	spaCy
Semantic Model	Sentence Transformers
Transformer Model	all-MiniLM-L6-v2
Machine Learning	Scikit-learn
Regression Model	Ridge Regression
Model Serialization	Joblib
Numerical Computing	NumPy
Data Processing	Pandas
Deployment	Streamlit Community Cloud



📁 Project Structure
automatic_assessment_deploy/
│
├── app.py
│
├── config/
│   ├── __init__.py
│   └── config.py
│
├── models/
│   ├── ridge_model.pkl
│   └── scaler.pkl
│
├── modules/
│   │
│   ├── dataset_split.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocess.py
│   │
│   ├── semantic_analysis/
│   │   ├── __init__.py
│   │   └── semantic_encoder.py
│   │
│   ├── syntactic_analysis/
│   │   ├── __init__.py
│   │   └── syntax_analyzer.py
│   │
│   ├── scoring/
│   │   ├── __init__.py
│   │   └── scoring_model.py
│   │
│   ├── explanation/
│   │   ├── __init__.py
│   │   └── explanation_module.py
│   │
│   └── pipeline/
│       ├── __init__.py
│       └── training_pipeline.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
│
├── requirements.txt
├── deployment_requirements.txt
├── README.MD
└── .gitignore
📦 Model Artifacts

The trained machine-learning artifacts are included in the project:

models/
├── ridge_model.pkl
└── scaler.pkl

These files are required by the deployed application.

The application loads these models using project-relative paths.

⚙️ Installation
1. Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd automatic_assessment_deploy
2. Create a Virtual Environment
python -m venv venv
Linux / WSL
source venv/bin/activate
Windows
venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
▶️ Run Locally

Start the Streamlit application:

streamlit run app.py

The application will normally be available at:

http://localhost:8501
🖥️ Application Workflow
Step 1 — Enter Question

Provide the essay or descriptive question.

Step 2 — Enter Student Answer

Enter the student's response.

Step 3 — Evaluate

Click:

✨ Evaluate Answer
Step 4 — View Assessment

The system displays:

Overall score
Content score
Language score
Organisation score
Overall performance
Assessment insights
Step 5 — View Explanation

An expandable explanation section provides assessment reasoning.

Step 6 — View Feedback

An expandable feedback section provides suggestions for improving the response.

🌐 Deployment

The application is deployed using:

Streamlit Community Cloud

Live Application

👉 https://autograde-ai.streamlit.app/

The deployed application runs without requiring:

WSL
CUDA
Dedicated GPU
Local GPU configuration
Large generative language models

The deployment uses the lightweight MiniLM semantic model together with the trained scoring model and explanation/feedback module.

🔒 Deployment Design

During development, large generative models were explored for explanation generation.

However, loading Phi-2 caused significant memory usage and environment instability during development.

Therefore, the deployed application intentionally avoids large generative models and GPU-specific dependencies.

The current lightweight deployment architecture is:

MiniLM Semantic Analysis
          +
Syntactic NLP Features
          +
Feature Fusion
          +
Ridge Regression
          +
Rubric-Based Assessment
          +
Explanation & Feedback

This makes the application suitable for lightweight cloud deployment and academic demonstration.

📈 Example Output

For a successfully evaluated response, AutoGrade AI can produce an assessment similar to:

Overall Score
8.08 / 10


Content
4.04 / 5


Language
1.62 / 2


Organisation
2.42 / 3

The exact score depends on the submitted question and answer.

🧪 Testing

The application has been tested with multiple types of questions and student responses.

Testing included:

Different question types
Relevant answers
Less relevant answers
Well-written answers
Short answers
Different response structures

The complete evaluation workflow was verified for:

Score calculation
Semantic similarity
Rubric-wise scores
Explanation generation
Feedback generation
Streamlit UI rendering
📚 Dataset

The project contains Automatic Essay Assessment dataset files and processed training/testing data.

Dataset-related files are organized under:

data/
├── raw/
├── processed/
└── outputs/

The repository also contains prediction and explanation outputs generated during development and evaluation.

🔬 Academic Context

AutoGrade AI was developed as an academic Automatic Answer Assessment and Feedback Generation system.

The primary objective is to explore how Natural Language Processing, transformer-based semantic representations, linguistic features, and machine learning can be combined to automate descriptive-answer evaluation.

The system integrates:

Semantic Understanding
        +
Linguistic Analysis
        +
Machine Learning
        +
Rubric-Based Assessment
        +
Automated Feedback
🚧 Future Improvements

Potential future enhancements include:

Support for multiple subject domains
Larger and more diverse training datasets
Improved rubric calibration
Advanced explainability techniques
Teacher/admin dashboards
Batch answer evaluation
Assessment history
Exportable assessment reports
Additional transformer models
Multilingual answer assessment
👩‍💻 Author

Sneha Sivakumar

Master of Computer Applications (MCA)

Areas of Interest
Artificial Intelligence
Machine Learning
Natural Language Processing
AI Evaluation
LLM Evaluation
Python
Intelligent Educational Systems
⭐ Project

If you find this project interesting, feel free to explore the repository and try the live application.

🚀 Live Demo

https://autograde-ai.streamlit.app/

🧠 AutoGrade AI

Automatic Answer Assessment & Intelligent Feedback System