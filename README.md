# AI Resume Screener with Streamlit

## Project Overview
This project develops an AI-powered resume screening application using Streamlit. The application allows users to upload PDF resumes or paste resume text, which is then processed to predict the most suitable job category using a trained machine learning model. The user interface is designed with an aesthetic inspired by Apple iOS for a clean and modern look.

## Features
-   **Resume Text Preprocessing**: Cleans and normalizes resume text by removing URLs, emails, special characters, stopwords, and performing lemmatization.
-   **TF-IDF Feature Extraction**: Converts cleaned resume text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).
-   **Random Forest Classification**: Utilizes a pre-trained Random Forest Classifier to predict job categories.
-   **PDF Resume Upload**: Allows users to upload PDF files, from which text is automatically extracted for analysis.
-   **Manual Text Input**: Provides a text area for users to paste resume content directly.
-   **Intuitive UI**: A Streamlit application with a clean, iOS-inspired design for an enhanced user experience.

## Dataset
The model was trained on the `Resume.csv` dataset, sourced from Kaggle, containing resume text and corresponding job categories. The dataset includes various categories such as 'HR', 'INFORMATION-TECHNOLOGY', 'FINANCE', 'ENGINEERING', and more.

## Model Details
-   **Algorithm**: Random Forest Classifier
-   **Feature Engineering**: TF-IDF Vectorization (`max_features=5000`)
-   **Performance**: Achieved an accuracy of approximately 68.61% on the test set, outperforming a Logistic Regression model (65.39% accuracy).
-   **Saved Components**: The trained `random_forest_model.joblib`, `tfidf_vectorizer.joblib`, and `label_encoder.joblib` are saved for direct use in the Streamlit application.

## Setup and Installation
To run this project, you'll need to set up your Python environment and install the necessary libraries.

### 1. Clone the repository (if applicable) or ensure files are in your Colab environment
If running in Colab, ensure all generated `.joblib` and `.csv` files (e.g., `random_forest_model.joblib`, `tfidf_vectorizer.joblib`, `label_encoder.joblib`) are in your working directory.

### 2. Install Dependencies
Install the required Python packages using `pip`. You can use the `requirements.txt` file generated earlier:

```bash
pip install -r requirements.txt
```

Alternatively, you can install them individually:
```bash
pip install pandas numpy scikit-learn nltk joblib streamlit pypdf kagglehub streamlit-app-runner ngrok
```

### 3. Download NLTK Data
Ensure the necessary NLTK data (stopwords, wordnet, punkt) are downloaded. This is handled automatically within the Streamlit app code but can be done manually if needed:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

## Running the Streamlit Application

To run the Streamlit application and access it via a public URL (e.g., when using Google Colab):

### 1. Save the Streamlit Code to `app.py`
Copy the entire Streamlit application code (from the cell that defined the `st.set_page_config`, `st.title`, etc.) into a file named `app.py` in your Colab environment's file system. You can do this manually using the Colab file browser:

-   Click the folder icon on the left sidebar.
-   Click `+ File` to create a new file and name it `app.py`.
-   Paste the Streamlit code into `app.py` and save.

### 2. Run the Application
Execute the following commands in a Colab code cell:

```python
!pip install ngrok streamlit-app-runner
from streamlit_app_runner import run_streamlit_app

# Ensure your app.py file is in the current directory or specify its path
run_streamlit_app('app.py')
```

This will start the Streamlit application and provide a public `ngrok` URL that you can open in your browser to interact with the app.

## Future Improvements
1.  **Hyperparameter Tuning**: Optimize the Random Forest model further (e.g., using GridSearchCV or RandomizedSearchCV).
2.  **Address Class Imbalance**: Implement techniques like SMOTE or adjust class weights for categories with low representation.
3.  **Explore Other Models**: Experiment with advanced models such as Gradient Boosting, SVMs, or deep learning architectures.
4.  **Feature Engineering**: Investigate different TF-IDF parameters, N-grams, or incorporate word embeddings (e.g., Word2Vec, GloVe) for richer text representation.
5.  **Error Analysis**: Conduct a deeper analysis of misclassified resumes to understand patterns and improve model robustness.
6.  **More Robust PDF Parsing**: Implement advanced error handling or alternative libraries for PDF text extraction.
7.  **Skill Extraction & Matching**: Add functionality to extract specific skills and match them against job descriptions.