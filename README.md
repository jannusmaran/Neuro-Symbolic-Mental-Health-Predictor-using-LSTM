#  Neuro-Symbolic Mental Health Predictor

## Overview
This project is a Deep Learning-based NLP system that analyzes textual data and classifies it into four emotional categories:
-  Happy  
-  Sad  
-  Stress  
-  Anxiety  

The goal is to identify potential mental health risks from user-generated text using an LSTM model.

---

##  Objective
- Build a deep learning model to classify emotions from text
- Simplify 28 emotions into 4 major categories
- Provide real-time predictions using a web application
- Assist in early detection of mental health conditions

---

## Tech Stack
- **Programming Language:** Python  
- **Libraries:**  
  - Pandas, NumPy  
  - TensorFlow / Keras  
  - Scikit-learn  
  - NLTK  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment:** Streamlit  
- **Model Saving:** Pickle  

---

##  Dataset
- **GoEmotions Dataset**
- Contains 28 emotion labels
- Multi-label dataset converted into single-label classification
- Grouped into 4 categories:
  - Happy
  - Sad
  - Stress
  - Anxiety

---

##  Methodology
1. Data Cleaning & Preprocessing  
2. Emotion Mapping (28 → 4 categories)  
3. Tokenization using Keras  
4. Text to Sequences conversion  
5. Padding sequences  
6. Model Building using LSTM  
7. Model Evaluation  

---

## Model Details
- Model Type: LSTM (Long Short-Term Memory)
- Input: Tokenized & padded text sequences
- Output: Emotion classification (4 classes)

---

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

---

## Results
- Achieved approximately **80% accuracy**
- Good performance on clear emotional texts
- Some confusion between similar emotions (Stress vs Anxiety)

---

## Challenges
- Handling multi-label dataset  
- Class imbalance  
- Similar emotion overlap  
- Consistent preprocessing  

---

## Real-World Applications
- Mental health monitoring systems  
- Emotion-aware chatbots  
- Social media sentiment analysis  
- Early detection of psychological distress  

---

## Future Improvements
- Use Transformer models (BERT, RoBERTa)
- Improve dataset balancing
- Hyperparameter tuning
- Real-time API deployment
- Add personalized suggestions

---

##  How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
