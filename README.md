BertAnalyzer



Overview

This project focuses on building a document classification model using BERT, a transformer-based model from Hugging Face. The model is designed to classify various types of documents such as invoices, shipping orders, and reports. Users can upload PDF files, and the model will predict their respective categories based on the extracted text.

Features

Automated Document Classification: Classifies documents into predefined categories.

PDF Upload Support: Users can upload PDFs for real-time classification.

Text Preprocessing: Cleans and processes text data for optimal model performance.

Fine-Tuned BERT Model: Utilizes a pre-trained BERT model adapted for document classification.

User Interface for Predictions: A simple UI for document uploads and classification results.

Data Preparation & Cleaning

The dataset consists of various document types.

Text cleaning includes removing special characters and stop words using Regular Expressions and NLTK.

A Label Encoder is applied to convert categorical labels (e.g., invoice, shipping order) into numerical format.

Model Fine-Tuning Process

BERT is fine-tuned on the custom dataset with tokenization and padding for uniform input size.

The dataset is split into 80% training and 20% testing.

Evaluation metrics such as precision, recall, and accuracy are used to assess performance.

Deployment & Usage

Model Saving: The trained model and label encoder are saved for future predictions.

Prediction System: Users can upload PDFs, and the system extracts text to classify the document.

UI Integration: A simple web interface allows users to upload PDFs and view classification results along with a preview of the document.

How to Use

Clone the repository:

git clone https://github.com/your-repo/document-classification-bert.git
cd document-classification-bert

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

Upload a PDF file via the UI and view classification results.

Conclusion

This project effectively demonstrates the use of BERT for document classification, covering steps from data preparation to deployment. The model’s accuracy and real-world applicability highlight the potential of transformer models in document processing tasks.

For further exploration, refer to additional resources in the documentation or repository.

Contributors: Your NameLicense: MITRepository: GitHub Link

