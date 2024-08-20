import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, logging
from huggingface_hub import login

# Ensure you have logged in to Hugging Face
hf_token = "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"
login(token=hf_token)

# Verify CUDA and GPU availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure CUDA is installed and the GPU drivers are correctly configured.")

print(f"Using GPU: {torch.cuda.get_device_name(0)}")

modelName = "google/gemma-2-2b"

bnbConfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Set the logging level to INFO to avoid excessive logs
logging.set_verbosity_info()

try:
    tokenizer = AutoTokenizer.from_pretrained(modelName)
except ValueError as e:
    print(f"Failed to load the tokenizer: {e}")
    raise

try:
    model = AutoModelForCausalLM.from_pretrained(
        modelName,
        device_map="auto",
        quantization_config=bnbConfig
    )
except Exception as e:
    print(f"Failed to load the model: {e}")
    raise

# Example prompt to test the model
system = "You are a skilled software architect who consistently creates system designs for various applications."
user = "Design a system with the ASCII diagram for the customer support application."

prompt = f"System: {system} \n User: {user} \n AI: "

inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

try:
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=500, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text.split("AI:")[1])
except Exception as e:
    print(f"Failed to generate text: {e}")
