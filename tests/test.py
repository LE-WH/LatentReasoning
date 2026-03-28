from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "tergel/qwen2.5-3b-instruct-gsm8k-fs-gpt4o-bon"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16)

question = "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take? Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format."

inputs = tokenizer(question, return_tensors="pt").to(device)
input_length = len(inputs['input_ids'][0])

outputs = model.generate(**inputs, max_new_tokens=512)

response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
print(response)