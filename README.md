# BertAnalyzer

## Overview
BertAnalyzer is a document classification system leveraging BERT, a transformer-based model from Hugging Face. This project enables users to upload PDF files, extracts text, and classifies them into categories such as invoices, shipping orders, and reports.

## Features
- **Automated Document Classification**: Classifies documents into predefined categories.
- **PDF Upload Support**: Users can upload PDFs for real-time classification.
- **Text Preprocessing**: Cleans and processes text data for optimal model performance.
- **Fine-Tuned BERT Model**: Utilizes a pre-trained BERT model adapted for document classification.
- **User-Friendly Interface**: A simple UI for document uploads and classification results.

## Data Preparation & Cleaning
- The dataset includes various document types.
- Text preprocessing involves removing special characters and stop words using Regular Expressions and NLTK.
- A Label Encoder converts categorical labels (e.g., invoice, shipping order) into numerical format for model compatibility.

## Model Fine-Tuning Process
- BERT is fine-tuned on the custom dataset using tokenization and padding for uniform input size.
- The dataset is split into 80% training and 20% testing.
- Evaluation metrics such as precision, recall, and accuracy assess model performance.

## Deployment & Usage
### Model Saving & Prediction System
- The trained model and label encoder are saved for future predictions.
- Users can upload PDFs, and the system extracts text for classification.

### UI Integration
- A simple web interface allows users to upload PDFs and view classification results.
- The uploaded document is previewed alongside its classification.

## How to Use
### Clone the repository:
```bash
git clone https://github.com/your-repo/document-classification-bert.git
cd document-classification-bert
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the application:
```bash
streamlit run app.py
```

### Upload a PDF file via the UI and view classification results.

## Conclusion
This project effectively demonstrates the use of BERT for document classification, covering steps from data preparation to deployment. The model’s accuracy and real-world applicability highlight the potential of transformer models in document processing tasks.

For further exploration, refer to additional resources in the repo
