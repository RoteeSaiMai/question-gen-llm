from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the GPT-2 model and tokenizer
model_name = "gpt2"  # or use "gpt2-medium", "gpt2-large", etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate multiple-choice questions
def generate_mcq_questions(text, num_questions):
    prompt = (
        f"Create {num_questions} multiple-choice questions from the following text in Thai:\n\n"
        f"Text: {text}\n\n"
        "Questions and Answers:"
    )
    
    # Generate text
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
    outputs = model.generate(**inputs, max_length=1500, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

# Example usage:
text_input = "ใส่ข้อความยาวๆ ที่นี่"
num_questions = 5
questions = generate_mcq_questions(text_input, num_questions)

print(questions)
