from PIL import Image
import os

def open_images_from_directory(directory):
    # List all files in the directory
    files = os.listdir(directory)
    img = []
    for file in (files):
        # Check if the file is an image
        if file.endswith('.jpg'):
            try:
                image_path = os.path.join(directory, file)
                img.append(Image.open(image_path))
            except Exception as e:
                print(f"Error opening image {file}: {e}")
    return img

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
DEVICE = "cuda" if torch.cuda.is_available() else "CPU"
print("DEVICE:", DEVICE)

def create_input_messages(image_array):
    # Generate image content based on the length of image_array
    image_contents = [{"type": "image"} for _ in image_array]
    
    # Add the text message to describe the images
    text_content = {"type": "text", "text": "Analyze this sequence of frames where the red spot shows the user's eye gaze. Identify whether the user is performing a pick or place task. If the task is picking, specify the item being picked up. If the task is placing, describe what is being placed and its destination."}
    
    # Combine image contents and text content
    messages = [{"role": "user", "content": image_contents + [text_content]}]
    
    return messages


processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/Idefics3-8B-Llama3", torch_dtype=torch.bfloat16
).to(DEVICE)

#CUDA out of memory with 3 images
directory = "/home/ttyh/hot3d/hot3d/dataset/Labelled/Videos/new_frames/Pick up_P0001_a68492d5_new_8"

img = open_images_from_directory(directory)
img = img[1:]

messages = create_input_messages(img)
print(messages)

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=img, return_tensors="pt")
#inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
inputs = inputs.to(DEVICE)

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=100)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)

'''
[{'role': 'user', 'content': [{'type': 'image'}, {'type': 'image'}, {'type': 'image'}, {'type': 'text', 'text': "Analyze this sequence of frames where the red spot shows the user's eye gaze. Identify whether the user is performing a pick or place task. If the task is picking, specify the item being picked up. If the task is placing, describe what is being placed and its destination."}]}]
["User:<image>Analyze this sequence of frames where the red spot shows the user's eye gaze. Identify whether the user is performing a pick or place task. If the task is picking, specify the item being picked up. If the task is placing, describe what is being placed and its destination.\nAssistant: The task is placing. The user is putting a milk carton on the shelf."]

'''