import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TextInput(BaseModel):
    text1: str
    text2: str

class BatchTextInput(BaseModel):
    text_pairs: List[dict]

class ModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load model information from environment variables or use local path
        model_path = os.getenv('MODEL_PATH', 'Z:/finetuned')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def predict(self, text1: str, text2: str) -> dict:
        inputs = self.tokenizer(
            text1,
            text2,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            confidence = probabilities[0][prediction.item()].item()

        return {
            "text1": text1,
            "text2": text2,
            "is_paraphrase": bool(prediction.item()),
            "confidence": round(confidence * 100, 2)
        }

    def predict_batch(self, text_pairs: List[dict]) -> List[dict]:
        return [self.predict(pair["text1"], pair["text2"]) for pair in text_pairs]

app = FastAPI(title="Text Analysis API")
model_service = None

@app.on_event("startup")
async def startup_event():
    global model_service
    model_service = ModelService()

@app.post("/predict")
async def predict(input_data: TextInput):
    try:
        result = model_service.predict(input_data.text1, input_data.text2)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(input_data: BatchTextInput):
    try:
        results = model_service.predict_batch(input_data.text_pairs)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def create_gradio_interface():
    def predict_texts(text1, text2):
        result = model_service.predict(text1, text2)
        return (
            f"Result: {'Similar' if result['is_paraphrase'] else 'Different'}\n"
            f"Confidence: {result['confidence']}%"
        )

    demo = gr.Interface(
        fn=predict_texts,
        inputs=[
            gr.Textbox(lines=4, placeholder="Enter first text...", label="Text 1"),
            gr.Textbox(lines=4, placeholder="Enter second text...", label="Text 2")
        ],
        outputs=gr.Textbox(label="Result"),
        title="Text Similarity Analysis",
        description="Compare two texts to check if they express similar meanings.",
        examples=[
            ["The cat is sitting on the mat.", "A cat rests on the mat."],
            ["The weather is beautiful today.", "It's raining heavily outside."],
            ["She loves reading books.", "Reading books is her passion."]
        ],
        theme=gr.themes.Soft()
    )
    return demo

gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)