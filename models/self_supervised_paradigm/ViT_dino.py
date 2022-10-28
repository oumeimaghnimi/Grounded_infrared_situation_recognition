import os
os.system('pip install git+https://github.com/huggingface/transformers.git --upgrade')

import gradio as gr
from transformers import ViTFeatureExtractor, ViTModel
import torch
import matplotlib.pyplot as plt

torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cats.jpg')

def get_attention_maps(pixel_values, attentions, nh):
  threshold = 0.6
  w_featmap = pixel_values.shape[-2] // model.config.patch_size
  h_featmap = pixel_values.shape[-1] // model.config.patch_size

  # we keep only a certain percentage of the mass
  val, idx = torch.sort(attentions)
  val /= torch.sum(val, dim=1, keepdim=True)
  cumval = torch.cumsum(val, dim=1)
  th_attn = cumval > (1 - threshold)
  idx2 = torch.argsort(idx)
  for head in range(nh):
      th_attn[head] = th_attn[head][idx2[head]]
  th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
  # interpolate
  th_attn = torch.nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=model.config.patch_size, mode="nearest")[0].cpu().numpy()

  attentions = attentions.reshape(nh, w_featmap, h_featmap)
  attentions = torch.nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=model.config.patch_size, mode="nearest")[0].cpu()
  attentions = attentions.detach().numpy()

  # save attentions heatmaps and return list of filenames
  output_dir = '.'
  os.makedirs(output_dir, exist_ok=True)
  attention_maps = []
  for j in range(nh):
      fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
      # save the attention map
      plt.imsave(fname=fname, arr=attentions[j], format='png')
      # append file name
      attention_maps.append(fname)

  return attention_maps

feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/dino-vits8", do_resize=False)
model = ViTModel.from_pretrained("facebook/dino-vits8", add_pooling_layer=False)

def visualize_attention(image):
  # normalize channels
  pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values 

  # forward pass
  outputs = model(pixel_values, output_attentions=True, interpolate_pos_encoding=True)

  # get attentions of last layer
  attentions = outputs.attentions[-1] 
  nh = attentions.shape[1] # number of heads

  # we keep only the output patch attention
  attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

  attention_maps = get_attention_maps(pixel_values, attentions, nh)
  
  return attention_maps

title = "Interactive demo: DINO"
description = "Demo for Facebook AI's DINO, a new method for self-supervised training of Vision Transformers. Using this method, they are capable of segmenting objects within an image without having ever been trained to do so. This can be observed by displaying the self-attention of the heads from the last layer for the [CLS] token query. This demo uses a ViT-S/8 trained with DINO. To use it, simply upload an image or use the example image below. Results will show up in a few seconds."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2104.14294'>Emerging Properties in Self-Supervised Vision Transformers</a> | <a href='https://github.com/facebookresearch/dino'>Github Repo</a></p>"
examples =[['cats.jpg']]

iface = gr.Interface(fn=visualize_attention, 
                     inputs=gr.inputs.Image(shape=(480, 480), type="pil"), 
                     outputs=[gr.outputs.Image(type='file', label=f'attention_head_{i}') for i in range(6)],
                     title=title,
                     description=description,
                     article=article,
                     examples=examples)
iface.launch()
