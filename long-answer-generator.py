# -*- coding: utf-8 -*-

#คำถามแบบยาวมีหลายคะแนนต่อข้อ
import ollama
import json
import re

def generate_question_prompt(previous_questions, text):
    """
    Generates a prompt for asking an open-ended question, considering previously asked questions.
    """
    # Start with the initial text in Thai
    prompt = f'ข้อความ: {text}\n\n'

    # Ask for a new question in Thai, strictly based on the provided text
    prompt += '''กรุณาสร้างคำถามอัตนัยหนึ่งข้อเป็นภาษาไทย โดยอิงจากข้อมูลและรายละเอียดในข้อความข้างต้นเท่านั้น. Don't ask about the text itself, just use the information in the text to form your question. Make sure the question is not ambiguous and has a clear answer, and that it is not open-ended and not left to interpretation. Try to incorporate elements of Bloom's taxonomy to dictate the structure of your questions.
    
    Bloom's Taxonomy:
    1. Remembering: Recalling facts and basic concepts (e.g., Define, List, Identify).
    2. Understanding: Explaining ideas or concepts (e.g., Explain, Summarize, Describe).
    3. Applying: Using information in new situations (e.g., Use, Implement, Solve).
    4. Analyzing: Drawing connections among ideas (e.g., Compare, Contrast, Examine).
    5. Evaluating: Justifying a decision or course of action (e.g., Judge, Critique, Recommend).
    6. Creating: Producing new or original work (e.g., Design, Assemble, Construct).

    Always start the question with [score] encasing the score with square brackets. Make sure to phrase your question like a Thai native. Make sure the response only contains the question text and nothing else; don't even add a question number or any other comments.
    '''

    # Include previously asked questions if any
    if previous_questions:
        prompt += "ด้านล่างเป็นรายการคำถามที่เคยถามไปแล้ว Don't ask these questions again:\n\n"
        for idx, question in enumerate(previous_questions, 1):
            prompt += f"{idx}. {question}\n"

    return prompt

def evaluate_response_prompt(question, user_response, previous_evaluations, text):
    """
    Generates a prompt for evaluating the user's response to a well-structured question.
    The evaluation should follow strict principles for assessment, providing clear reasoning and constructive feedback.
    """
    prompt = f'ข้อความ: {text}\n\n'
    prompt += f"คำถามเกี่ยวกับข้อความที่ถูกถามไป: {question}\n\n"
    prompt += f"คำตอบของผู้ใช้:\n{user_response}\n\n"
    prompt += '''จากคำตอบนี้ กรุณาประเมินคำตอบและให้ข้อเสนอแนะโดยใช้หลักการที่ดีในการประเมินคำตอบ โดยเน้นความถูกต้องของข้อมูล การใช้เหตุผล และความสมบูรณ์ของคำตอบ 

    Do not give out marks unless answer is exactly correct. somthing like a close number or a tangentially related answer should still get 0.
    
    โปรดจำคะแนน [score] จากคำถามและให้คำตอบมากที่สุดห้ามเกินเลขนี้ แป็นคะแนนเต็ม (ให้เป็นคะแนนเต็มที่เหมาะสมหากคำตอบสมบูรณ์ หรือให้คะแนนต่ำหากคำตอบไม่สมบูรณ์)
    โปรดให้คำตอบที่ถูกต้องพร้อมเหตุผลที่ชัดเจนและเฉพาะเจาะจงสำหรับการประเมินนั้น คำตอบควรอยู่ในขอบเขตไม่เกิน 4 ประโยค และใช้ภาษาไทยที่เข้าใจง่ายเท่านั้น.

    สำหรับการประเมินคำตอบ ให้แบ่งคะแนนออกเป็นส่วนๆโดยให้คำอธิบายของแต่ละคะแนนโดยที่คะแนนเต็มมาจากเลขที่อยู่ในคำถาม

    โปรดใช้รูปแบบการเริ่มต้นการประเมินดังนี้:
    - ระบุว่าคำตอบถูกต้องหรือผิด
    - อธิบายข้อผิดพลาดในคำตอบและให้คำตอบที่ถูกต้องพร้อมเหตุผล.
    - กรุณาระบุคะแนนสำหรับแต่ละหัวข้อ เช่น ความรับผิดชอบและความโปร่งใส การจัดการข้อมูล และความถูกต้องโดยรวม
    - เขียนคะแนนเป็น JSON ที่มีเฉพาะคะแนนสุดท้ายและคะแนนย่อยที่ถูกระบุ

    The json file should only have one column named "score"

    Make sure that the score given does not exceed the amount of marks detailed in the question.

    ให้คำตอบเป็นภาษาไทยเท่านั้นห้ามมีภาษาอื่น และให้คำตอบควรอยู่ในขอบเขตไม่เกิน 4 ประโยค
    Write your response in Thai only. DONT' ANSWER IN ENGILISH WHAT DID I TELL YOU WHY ARE YOU ANSWERING IN ENGLISH I SAID DON'T ANSWER IN ENGLISH.
    '''
    return prompt

def parse_evaluation_score(evaluation_text):
    """
    Parses the evaluation text to extract the score from the JSON format.
    """
    # Extract JSON score using a more flexible approach
    try:
        # Find the first occurrence of a JSON-like structure in the text
        json_match = re.search(r'\{.*\}', evaluation_text, re.DOTALL)
        if json_match:
            score_json = json_match.group(0)
            score_data = json.loads(score_json)
            return score_data.get("score", 0)  # Return the score from the JSON
        else:
            print("No JSON score found in evaluation text.")
            return 0
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing JSON: {e}")
        return 0

def ask_open_ended_questions(num_questions, initial_text):
    """
    Main function to ask open-ended questions and evaluate user responses.
    
    Parameters:
        num_questions (int): The number of open-ended questions to ask.
        initial_text (str): Initial context or text on which the questions are based.
        
    Returns:
        int: The total score (number of correct answers).
    """
    previous_questions = []
    previous_evaluations = []
    total_score = 0
    
    for i in range(num_questions):
        # Generate the next question
        question_prompt = generate_question_prompt(previous_questions, initial_text)
     #    print(f'Prompt: {question_prompt}')
        response = ollama.generate(model="llama3", prompt=question_prompt, stream=True)
        generated_text = "".join([chunk["response"] for chunk in response])
     #    print(f"Generated question text: {generated_text}")

        # Extract the new question
        new_question = generated_text.strip()
        previous_questions.append(new_question)
        
        # Ask the user for their response
        user_response = input(f"Question {i + 1}: {new_question}\nYour response: ")

        # Evaluate the response
        evaluation_prompt = evaluate_response_prompt(new_question, user_response, previous_evaluations, initial_text)
        evaluation_response = ollama.generate(model="llama3", prompt=evaluation_prompt, stream=True)
        evaluation_text = "".join([chunk["response"] for chunk in evaluation_response])
     #    print(f"Evaluation text: {evaluation_text}")

        # Parse the evaluation to extract the score
        score = parse_evaluation_score(evaluation_text)
        total_score += score

        # Keep track of the evaluation for future context
        previous_evaluations.append(evaluation_text.strip())
        
        # Provide feedback to the user
        print(f"Feedback for Question {i + 1}:\n{evaluation_text}\n")
        print(f"Score for Question {i + 1}: {score}\n")

    return total_score

# Example usage:
text_input = '''
ข้อพิจารณาทางจริยธรรมในการนำ AI มาใช้
การบรรเทาอคติและความเป็นธรรม
ข้อกังวลหลักด้านจริยธรรมประการหนึ่งใน AI เกี่ยวข้องกับอคติที่มีอยู่ในข้อมูลและอัลกอริธึม

การนำ AI มาใช้อย่างมีจริยธรรมจำเป็นต้องมีมาตรการเชิงรุกเพื่อลดอคติและรับประกันความยุติธรรม

อัลกอริทึมจะต้องได้รับการตรวจสอบและปรับเปลี่ยนอย่างต่อเนื่องเพื่อป้องกันการเกิดอคติทางสังคม การเลือกปฏิบัติ หรือแนวปฏิบัติที่กีดกันทางสังคม

ความรับผิดชอบและความโปร่งใส
การนำ AI มาใช้อย่างมีจริยธรรมต้องอาศัยความรับผิดชอบและความโปร่งใสในการพัฒนาและปรับใช้ระบบ AI

องค์กรควรให้ความกระจ่างเกี่ยวกับวิธีการใช้ AI กระบวนการตัดสินใจ และผลกระทบที่อาจเกิดขึ้น

ความโปร่งใสส่งเสริมความไว้วางใจระหว่างผู้ใช้และผู้มีส่วนได้ส่วนเสีย ช่วยให้ตัดสินใจโดยมีข้อมูลประกอบเกี่ยวกับการโต้ตอบของ AI

ความเป็นส่วนตัวและความปลอดภัยของข้อมูล
การปกป้องความเป็นส่วนตัวของผู้ใช้และความปลอดภัยของข้อมูลเป็นสิ่งสำคัญยิ่งในการนำ AI มาใช้อย่างมีจริยธรรม

องค์กรต่างๆ ต้องใช้มาตรการที่เข้มงวดเพื่อปกป้องข้อมูลที่ละเอียดอ่อน ปฏิบัติตามกฎข้อบังคับในการปกป้องข้อมูล และจัดลำดับความสำคัญของการยินยอมและการควบคุมข้อมูลของผู้ใช้

การจัดการข้อมูลอย่างมีความรับผิดชอบทำให้ผู้ใช้ไว้วางใจและรักษามาตรฐานทางจริยธรรม

ผลกระทบต่อสังคมและความรับผิดชอบ
การนำ AI มาใช้จำเป็นต้องคำนึงถึงผลกระทบต่อสังคมในวงกว้าง

การใช้งานอย่างมีความรับผิดชอบเกี่ยวข้องกับการทำความเข้าใจและการบรรเทาผลกระทบทางสังคมที่อาจเกิดขึ้นจากแอปพลิเคชัน AI

องค์กรควรมีส่วนร่วมอย่างแข็งขันต่อความเป็นอยู่ที่ดีของสังคม มีส่วนร่วมในการอภิปราย โครงการริเริ่ม และนโยบายที่ส่งเสริมการใช้ AI อย่างมีจริยธรรม และ
'''

num_questions = 2
total_score = ask_open_ended_questions(num_questions, text_input)

print(f"Total Score: {total_score}/{num_questions}")
