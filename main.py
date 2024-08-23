from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from adaptive_long_answer import generate_question_prompt_long_adaptive, evaluate_response_prompt_long_adaptive, adjust_difficulty
from long_answer import generate_question_prompt_long, evaluate_response_prompt_long, parse_evaluation_score
from MCQ_answer import generate_mcq_questions, extract_json_from_text, save_mcq_to_csv
import re

import ollama
from typing import Optional, List, Dict, Any

app = FastAPI()

# class InteractiveQuizRequest(BaseModel):
#     text: str = Field(
#         ...,
#         example="""หญ้า เป็นวงศ์ของพืชดอกใบเลี้ยงเดี่ยวที่มีจำนวนมากและมีแทบทุกหนแห่ง โดยมีประมาณ 780 สกุลและประมาณ 12,000 สปีชีส์...""",
#         description="Input text for generating questions. This field supports multi-line text."
#     )
#     num_questions: int = Field(
#         3,
#         example=3,
#         ge=1,
#         description="Number of questions to generate and evaluate."
#     )
#     initial_difficulty: int = Field(
#         3,
#         example=3,
#         ge=1, le=5,
#         description="Starting difficulty level for the questions."
#     )

# class InteractiveQuizResponse(BaseModel):
#     question: str
#     evaluation: str
#     score: int

# Define the data models with examples
class QuestionRequestAdaptive(BaseModel):
    text: str = Field(
        ...,
        example="""หญ้า เป็นวงศ์ของพืชดอกใบเลี้ยงเดี่ยวที่มีจำนวนมากและมีแทบทุกหนแห่ง โดยมีประมาณ 780 สกุลและประมาณ 12,000 สปีชีส์...""",
        description="Input text for generating questions. This field supports multi-line text."
    )
    previous_questions: list[str] = Field(
        default=[],
        example=["คำถามที่เคยถามไปก่อนหน้านี้"]
    )
    difficulty_level: int = Field(
        3,
        example=3,
        ge=1, le=5,
        description="Difficulty level for the question (1 to 5)."
    )

# Define the data models with examples
class QuestionRequest(BaseModel):
    text: str = Field(
        ...,
        example="""หญ้า เป็นวงศ์ของพืชดอกใบเลี้ยงเดี่ยวที่มีจำนวนมากและมีแทบทุกหนแห่ง โดยมีประมาณ 780 สกุลและประมาณ 12,000 สปีชีส์...""",
        description="Input text for generating questions. This field supports multi-line text."
    )
    previous_questions: list[str] = Field(
        default=[],
        example=["คำถามที่เคยถามไปก่อนหน้านี้"]
    )

class EvaluateRequest(BaseModel):
    question: str = Field(
        ...,
        example="ทุ่งหญ้ามีความสำคัญทางเศรษฐกิจอย่างไร?",
        description="The question being evaluated."
    )
    user_response: str = Field(
        ...,
        example="ทุ่งหญ้ามีความสำคัญเพราะสามารถนำธัญพืชมาใช้เป็นอาหารหลักหรือเป็นเชื้อเพลิงชีวภาพได้",
        description="The user's response to the question."
    )
    text: str = Field(
        ...,
        example="""หญ้า เป็นวงศ์ของพืชดอกใบเลี้ยงเดี่ยวที่มีจำนวนมากและมีแทบทุกหนแห่ง โดยมีประมาณ 780 สกุลและประมาณ 12,000 สปีชีส์...""",
        description="The original text from which the question was generated."
    )

# Define the data models with examples
class MCQRequest(BaseModel):
    text: str = Field(
        ...,
        example="""วิทยาการทางการแพทย์มีการพัฒนาและการคิดค้นสิ่งใหม่ๆ เกิดขึ้นอย่างต่อเนื่องเพื่อให้การทำงานของแพทย์ส่งผลที่ดีที่สุดให้กับผู้ป่วย...""",
        description="Input text for generating multiple-choice questions. This field supports multi-line text."
    )
    num_questions: int = Field(
        5,
        example=5,
        description="Number of multiple-choice questions to generate."
    )

class SaveMCQRequest(BaseModel):
    mcq_list: List[Dict[str, Any]]
    file_name: str = Field(
        "mcq_questions_gemma2b.csv",
        example="mcq_questions_gemma2b.csv",
        description="Filename to save the generated MCQ questions as a CSV."
    )


# Endpoint to generate MCQ questions
@app.post("/generate/mcq")
async def generate_mcq(data: MCQRequest):
    try:
        mcq_list = generate_mcq_questions(data.text, data.num_questions)
        return {"mcq_list": mcq_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating MCQ questions: {str(e)}")


# Endpoint to generate a question
@app.post("/generate/long/adaptive")
async def generate_question(data: QuestionRequestAdaptive):
    try:
        prompt = generate_question_prompt_long_adaptive(data.previous_questions, data.text, data.difficulty_level)
        response = ollama.generate(model="llama3", prompt=prompt, stream=True)
        generated_text = "".join([chunk["response"] for chunk in response])
        
        return {"question": generated_text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating question: {str(e)}")

# Endpoint to evaluate a response
@app.post("/evaluate/long/adaptive")
async def evaluate_response(data: EvaluateRequest):
    try:
        prompt = evaluate_response_prompt_long_adaptive(data.question, data.user_response, data.text)
        response = ollama.generate(model="llama3", prompt=prompt, stream=True)
        evaluation_text = "".join([chunk["response"] for chunk in response])
        score = parse_evaluation_score(evaluation_text)
        
        return {"evaluation": evaluation_text.strip(), "score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating response: {str(e)}")
    
# @app.post("/quiz/interactive", response_model=list[InteractiveQuizResponse])
# async def interactive_quiz(data: InteractiveQuizRequest):
#     try:
#         previous_questions = []
#         total_score = 0
#         difficulty_level = data.initial_difficulty
#         responses = []

#         for i in range(data.num_questions):
#             # Generate the next question
#             question_prompt = generate_question_prompt_long_adaptive(previous_questions, data.text, difficulty_level)
#             response = ollama.generate(model="llama3", prompt=question_prompt, stream=True)
#             generated_text = "".join([chunk["response"] for chunk in response])

#             # Extract the new question
#             new_question = generated_text.strip()
#             previous_questions.append(new_question)
            
#             # Here, you'd ask the user for their response
#             # For the sake of the example, let's assume we have a static response
#             user_response = "ตัวอย่างคำตอบของผู้ใช้"  # This would normally come from the user

#             # Evaluate the response
#             evaluation_prompt = evaluate_response_prompt_long_adaptive(new_question, user_response, data.text)
#             evaluation_response = ollama.generate(model="llama3", prompt=evaluation_prompt, stream=True)
#             evaluation_text = "".join([chunk["response"] for chunk in evaluation_response])

#             # Parse the evaluation to extract the score
#             score = parse_evaluation_score(evaluation_text)
#             total_score += score

#             # Adjust the difficulty for the next question
#             max_score = int(re.search(r'\[(\d+)\]', new_question).group(1))  # Extract the max score from the question
#             difficulty_level = adjust_difficulty(difficulty_level, score, max_score)

#             # Store the result for this question
#             responses.append(InteractiveQuizResponse(
#                 question=new_question,
#                 evaluation=evaluation_text.strip(),
#                 score=score
#             ))

#         return responses

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error during interactive quiz: {str(e)}")

# Endpoint to generate a question
@app.post("/generate/long")
async def generate_question(data: QuestionRequest):
    try:
        prompt = generate_question_prompt_long(data.previous_questions, data.text)
        response = ollama.generate(model="llama3", prompt=prompt, stream=True)
        generated_text = "".join([chunk["response"] for chunk in response])
        
        return {"question": generated_text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating question: {str(e)}")

# Endpoint to evaluate a response
@app.post("/evaluate/long")
async def evaluate_response(data: EvaluateRequest):
    try:
        prompt = evaluate_response_prompt_long(data.question, data.user_response, data.text)
        response = ollama.generate(model="llama3", prompt=prompt, stream=True)
        evaluation_text = "".join([chunk["response"] for chunk in response])
        score = parse_evaluation_score(evaluation_text)
        
        return {"evaluation": evaluation_text.strip(), "score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating response: {str(e)}")

# Example FastAPI main application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
