import torch
import gradio as gr
from PIL import Image
import requests

labels = ['calling',
 'clapping',
 'cycling',
 'dancing',
 'drinking',
 'eating',
 'fighting',
 'hugging',
 'laughing',
 'listening_to_music',
 'running',
 'sitting',
 'sleeping',
 'texting',
 'using_laptop']
 
 
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
  label2id[label] = i 
  id2label[id] = label 
  


from transformers import AutoModelForImageClassification, AutoFeatureExtractor
repo_name = "DrishtiSharma/finetuned-ViT-human-action-recognition-v1"


feature_extractor = AutoFeatureExtractor.from_pretrained(repo_name)
model = AutoModelForImageClassification.from_pretrained(repo_name)

def fn(image):
   image = Image.fromarray(image.astype('uint8'), 'RGB')
   encoding = feature_extractor(image.convert("RGB"), return_tensors = "pt")
   with torch.no_grad():
      outputs = model(**encoding)
      logits = outputs.logits
   predicted_class_idx = logits.argmax(-1).item()
   return f"predicted_class:, {model.config.id2label[predicted_class_idx]}"
   

gr.Interface(fn=fn, inputs = [gr.inputs.Image(label="Image to search", optional=True)],theme = "grass", outputs = gr.outputs.Textbox(), title = "Human Action Recognizer", examples = [["running.jpg"], ["sitting.jpg"],["jumping.jpg"]]).launch()
