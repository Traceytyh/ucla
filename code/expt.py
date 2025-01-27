#!/usr/bin/env python3
import requests
from PIL import Image


url = 'https://assets.dmagstatic.com/wp-content/uploads/2016/05/dogs-playing-in-grass.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')  
image.resize((596, 437)).show


#import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration



processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print('device: ', device)

#inputs = processor(image, return_tensors="pt").to(device, torch.float16)

#generated_ids = model.generate(**inputs, max_new_tokens=20)
#generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#print(generated_text)