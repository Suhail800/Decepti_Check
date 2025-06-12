# **DeceptiCheck: A Deception Detection Engine**

## **Unmasking Deceptive Content in Legal, Social, and Digital Texts with AI**

## **📝 Project Description**

DeceptiCheck is a robust and insightful AI-powered system designed to identify deceptive content within textual data. Developed as a capstone college project, it serves as a portfolio-worthy demonstration of advanced Natural Language Processing (NLP) and Machine Learning techniques applied to a critical real-world problem.  
While the project's broader vision encompasses legal, social, and digital texts, this iteration primarily focuses on detecting **deceptive opinion spam** within **social and digital texts, specifically online reviews**. It leverages a fine-tuned Transformer-based model to classify reviews as either "truthful" or "deceptive" and provides powerful explainability features to give users a deeper understanding of the model's predictions.

## **✨ Key Features**

- **Deception Prediction:** Classifies input text as Truthful or Deceptive with confidence scores.
- **Intuitive Web Interface:** A user-friendly Streamlit application for easy interaction and analysis.
- **Word Importance Highlighting:** Visually highlights words that most influenced the model's prediction, offering immediate insights into linguistic cues of deception.
- **Linguistic Feature Analysis:** Provides detailed statistics on various linguistic markers often associated with deceptive language (e.g., pronoun usage, readability scores).
- **Sentiment Analysis:** Integrates sentiment analysis to assess the emotional tone of the text, offering another layer of analytical depth.
- **Modern NLP Architecture:** Built upon a state-of-the-art Transformer model (DistilBERT) for high performance while remaining suitable for local execution.

## **🚀 Technical Stack**

- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Deep Learning Framework:** PyTorch
- **NLP Library:** Hugging Face Transformers
- **Data Manipulation:** Pandas
- **Text Analysis:** NLTK, TextStat
- **Numerical Operations:** NumPy
- **Plotting:** Matplotlib, Seaborn

## **📊 Dataset**

The model is trained on the **Deceptive Opinion Spam Dataset** (Ott et al., 2011/2013). This dataset consists of a collection of truthful and deceptive hotel reviews, making it an excellent resource for training models to distinguish genuine opinions from fabricated ones in a social/digital context.

## **🧠 Model**

**DistilBERT (distilbert-base-uncased)** is utilized as the core of the deception detection engine. DistilBERT is a smaller, faster, and lighter version of BERT, pre-trained on a large corpus of English data. It offers a powerful balance between cutting-edge performance and computational efficiency, making it ideal for running on local machines while still providing impressive accuracy. The model is fine-tuned on the aforementioned deceptive opinion spam dataset for binary text classification.

## **🔍 Explainability**

A core aspect of DeceptiCheck is its focus on explainable AI (XAI). The application provides:

- **Attention-based Highlighting:** By analyzing the attention weights from the DistilBERT model, the app highlights words the model focused on. If the prediction is deceptive, these words are marked in **red**; if truthful, they are marked in **green**.
- **Quantifiable Linguistic Cues:** Beyond the deep learning model, the app extracts and displays classic psycholinguistic features (e.g., frequency of first-person pronouns, readability scores, sentiment polarity), offering human-interpretable reasons behind the model's decision or generally observed patterns in deceptive language.

## **📂 Project Structure**

DeceptiCheck/  
├── app.py \# Main Streamlit web application  
├── model_training.py \# Script for training and saving the DistilBERT model  
├── utils.py \# Helper functions (data loading, preprocessing, linguistic features, prediction logic)  
├── requirements.txt \# List of Python dependencies  
├── data/ \# Directory for the dataset  
│ └── deceptive-opinion-spam-corpus/  
│ └── deceptive-opinion.csv \# The main dataset file  
└── models/ \# Directory to store the trained DistilBERT model  
 └── distilbert_deception_model/  
 ├── config.json  
 ├── pytorch_model.bin  
 ├── tokenizer.json  
 └── vocab.txt  
 └── ... (other tokenizer files)

## **🛠️ Setup and Local Installation (for VS Code / Local Environment)**

To set up and run DeceptiCheck on your local machine using VS Code, follow these steps:

1. **Clone the Repository:**  
   git clone https://github.com/YourGitHubUsername/DeceptiCheck.git  
   cd DeceptiCheck

   _(Replace https://github.com/YourGitHubUsername/DeceptiCheck.git with your actual repository URL)_

2. Open in VS Code:  
   Open the DeceptiCheck folder in Visual Studio Code. VS Code will often recommend installing Python extensions and setting up a virtual environment.
3. Create a Virtual Environment (Recommended):  
   Open the integrated terminal in VS Code (Terminal \> New Terminal) and run:  
   python \-m venv venv  
   \# On Windows:  
   .\\venv\\Scripts\\activate  
   \# On macOS/Linux:  
   source venv/bin/activate

   Ensure your VS Code is using this activated virtual environment (check the Python interpreter selection in the bottom-left status bar).

4. **Install Dependencies:**  
   pip install \-r requirements.txt

   Your requirements.txt should contain:  
   streamlit  
   pandas  
   scikit-learn  
   transformers  
   torch  
   nltk  
   textstat  
   matplotlib  
   seaborn  
   datasets  
   accelerate

5. Download NLTK Data:  
   Some components of NLTK (Natural Language Toolkit) are required. Run the following command in your activated virtual environment's terminal:  
   python \-c "import nltk; nltk.download('all')"

   This might take a few minutes as it downloads several large corpora.

6. Download the Dataset:  
   The project uses the "Deceptive Opinion Spam Corpus" available on Kaggle.
   - **Manual Download:** Go to [Kaggle: Deceptive Opinion Spam Corpus](https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus), log in, and download the dataset (deceptive-opinion-spam-corpus.zip).
   - Place the dataset: Unzip the downloaded file and place the deceptive-opinion-spam-corpus folder (containing deceptive-opinion.csv) into the data/ directory of your project.  
     Your path should look like: DeceptiCheck/data/deceptive-opinion-spam-corpus/deceptive-opinion.csv
7. Train the Deception Detection Model:  
   This step will fine-tune the DistilBERT model on your dataset. It will save the trained model artifacts in the models/distilbert_deception_model directory. Run this in your VS Code terminal:  
   python model_training.py

   This process can take some time depending on your hardware.

8. Run the Streamlit Application:  
   Once the model training is complete and the model is saved, you can launch the web interface. Run this in your VS Code terminal:  
   streamlit run app.py

   Your default web browser should automatically open to the Streamlit app (usually at http://localhost:8501).

## **🚀 Usage**

1. **Launch the App:** Follow the "Run the Streamlit Application" step above.
2. **Enter Text:** In the Streamlit interface, paste or type any social or digital text (ideally a review for best results) into the provided text area.
3. **Analyze:** Click the "Analyze Text for Deception" button.
4. **Review Results:**
   - See the model's prediction (Truthful/Deceptive) and confidence score.
   - Observe the "Word Importance" highlighting: words in red influenced a "deceptive" prediction, while green influenced a "truthful" one.
   - Examine the "Detailed Linguistic Features" and "Sentiment Analysis" for deeper insights into the text's characteristics.

## **💡 Future Enhancements**

This project can be extended in many exciting ways:

- **Advanced Explainability:** Implement LIME (Local Interpretable Model-agnostic Explanations) or SHAP (SHapley Additive exPlanations) for more granular and theoretically sound word attribution.
- **Broader Deception Domains:** Expand the dataset to include fake news articles, social media misinformation, or even curated legal text samples. This would likely require re-training the model on a more diverse corpus.
- **Multi-Modal Analysis:** Incorporate non-textual features, such as reviewer metadata (e.g., number of reviews, review history, ratings consistency) or image analysis if relevant to the source data.
- **User Feedback Loop:** Allow users to provide feedback on predictions, helping to refine the model over time.
- **Performance Optimization:** For production-level deployment, explore techniques like model quantization or ONNX export for faster inference.
- **Interactive Visualizations:** Develop more interactive charts and graphs within Streamlit to explore linguistic features across different text types.
