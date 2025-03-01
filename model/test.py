from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")
print(model)

state_dict = model.state_dict()

total_memory = 0
for name, param in state_dict.items():
    param_memory = param.numel() * param.element_size()  # numel() gives the number of elements, element_size() gives the size in bytes
    total_memory += param_memory

total_memory_mb = total_memory / (1024 * 1024)
print(f"Total memory usage of the state_dict: {total_memory_mb:.2f} MB")

# Total memory usage of the state_dict: 2858.13 MB




