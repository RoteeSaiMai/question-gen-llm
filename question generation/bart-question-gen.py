import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

# Check for GPU availability
# Check for GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load the GPT-2 model and tokenizer
model_name = "gpt2"  # You can use "gpt2-medium" or "gpt2-large" for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Initialize the pipeline for text generation
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

def generate_questions(text, num_questions=3):
    """
    Generates multiple-choice questions from a given text using the GPT-2 model.
    
    Args:
    - text (str): The input text to generate questions from.
    - num_questions (int): The number of questions to generate.

    Returns:
    - List[Dict[str, Any]]: A list of generated questions and possible answers.
    """
    prompt = (
        f"Generate {num_questions} multiple-choice questions from the following text in Thai. "
        "For each question, provide four answer choices labeled ก, ข, ค, ง and indicate the correct answer.\n\n"
        f"Text: {text}\n\nQuestions:"
    )

    # Generate questions
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(**inputs, max_length=512, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process to extract questions and answers
    questions_answers = generated_text.split("Questions:")[1].strip().split('\n\n')

    mcq_list = []
    for qa in questions_answers:
        if not qa.strip():
            continue

        lines = qa.split('\n')
        if len(lines) >= 6:
            question = lines[0].replace("Question:", "").strip()
            choices = {
                "ก": lines[1].replace("ก:", "").strip(),
                "ข": lines[2].replace("ข:", "").strip(),
                "ค": lines[3].replace("ค:", "").strip(),
                "ง": lines[4].replace("ง:", "").strip(),
            }
            answer = lines[5].replace("Answer:", "").strip()

            mcq_list.append({
                "question": question,
                "choices": choices,
                "answer": answer
            })

    return mcq_list

# Example usage:
text_input = """
หญ้า เป็นวงศ์ของพืชดอกใบเลี้ยงเดี่ยวที่มีจำนวนมากและมีแทบทุกหนแห่ง โดยมีประมาณ 780 สกุลและประมาณ 12,000 สปีชีส์
ทำให้หญ้าเป็นวงศ์พืชที่ใหญ่เป็นอันดับ 5 โดยเป็นรองเพียงวงศ์ทานตะวัน, วงศ์กล้วยไม้, วงศ์ถั่ว และวงศ์เข็ม.
"""
num_questions = 3
questions = generate_questions(text_input, num_questions)

# Print the generated questions
for i, mcq in enumerate(questions, 1):
    print(f"Question {i}: {mcq['question']}")
    print(f"ก: {mcq['choices']['ก']}")
    print(f"ข: {mcq['choices']['ข']}")
    print(f"ค: {mcq['choices']['ค']}")
    print(f"ง: {mcq['choices']['ง']}")
    print(f"Answer: {mcq['answer']}\n")
