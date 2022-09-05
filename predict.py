from transformers import AdamW, get_linear_schedule_with_warmup
import transformers
import torch
from torch import nn, optim
from torch.utils import data
from ModelClasses import BERTurkSentimentAnalyzer, ELECTRASentimentAnalyzer
from fastapi import FastAPI, Form, Query, HTTPException, status, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import random_generator

# MODEL_NAME = "dbmdz/bert-base-turkish-cased"
class_names = ['Negative', 'Positive', 'Neutral']
model = ELECTRASentimentAnalyzer(class_count=3)
model_path = 'models/best_model_state_electra.bin'
model.load_state_dict(torch.load(model_path, map_location='cpu'))
tokenizer = transformers.AutoTokenizer.from_pretrained("dbmdz/electra-base-turkish-cased-discriminator")

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.absolute() / "static"),
    name="static",
)

templates = Jinja2Templates(directory="templates")

class GivenText(BaseModel):
    text: str

def encoder(text):
    encoded_text = tokenizer.encode_plus(
        text,
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    return encoded_text

def predict(sample_text):
    model.eval()

    encoded_text = encoder(sample_text)
    input_ids = encoded_text['input_ids'].to("cpu")
    attention_mask = encoded_text['attention_mask'].to("cpu")

    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)

    return output

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/get_random_input")
def get_random_input(request: Request):
    return random_generator.generate_text()

@app.post("/predict")
async def predict_text(request: Request, text: GivenText):
    output = predict(text.text)
    print(request.body)

    neg = output[0][0].item()
    pos = output[0][1].item()
    neu = output[0][2].item()

    print(neg, pos, neu)
    
    # return {'Negative': neg, 
    #         'Positive': pos, 
    #         'Neutral': neu}

    return {"text": text,
            "neg": str("%.2f" %(neg*100)), 
            "pos": str("%.2f" %(pos*100)), 
            "neu": str("%.2f" %(neu*100))}

