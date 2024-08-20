from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from question_evaluation import generate_question_prompt, evaluate_response_prompt
import ollama

app = FastAPI()

# Define the data models with examples
class QuestionRequest(BaseModel):
    text: str = Field(
        ...,
        example="""หญ้า เป็นวงศ์ของพืชดอกใบเลี้ยงเดี่ยวที่มีจำนวนมากและมีแทบทุกหนแห่ง โดยมีประมาณ 780 สกุลและประมาณ 12,000 สปีชีส์[4] ทำให้หญ้าเป็นวงศ์พืชที่ใหญ่เป็นอันดับ 5 โดยเป็นรองเพียงวงศ์ทานตะวัน, วงศ์กล้วยไม้, วงศ์ถั่ว และวงศ์เข็ม[5]

หญ้าเป็นวงศ์พืชที่มีความสำคัญทางเศรษฐกิจมากที่สุด โดยสามารถนำธัญพืช เช่น ข้าวโพด, ข้าวสาลี, ข้าว, ข้าวบาร์เลย์ และข้าวฟ่าง ไปผลิตอาหารหลักหรือให้อาหารสัตว์ที่ผลิตเนื้อได้ พวกมันให้พลังงานทางอาหารมากกว่าครึ่งหนึ่ง (51%) ของพลังงานทางอาหารทั้งหมด ผ่านการบริโภคของมนุษย์โดยตรง แบ่งเป็นข้าว 20%, ข้าวสาลี 20%, ข้าวโพด 5.5% และธัญพืชอื่น ๆ 6%[6]

สมาชิกวงศ์หญ้าบางส่วนใช้เป็นวัสดุก่อสร้าง (ไม้ไผ่, มุงจาก และฟาง) ในขณะที่บางส่วนเป็นเชื้อเพลิงชีวภาพ ผ่านการแปลงข้าวโพดเป็นเอทานอล

ทุ่งหญ้าอย่างสะวันนาและแพรรีที่มีหญ้าเป็นส่วนใหญ่ ประมาณการว่าครอบคลุมไปถึง 40.5% ของพื้นที่ผิวโลก (ไม่นับกรีนแลนด์และแอนตาร์กติกา)[7]

หญ้ายังมีส่วนสำคัญต่อพืชพรรณในสภาพแวดล้อมต่าง ๆ ซึ่งรวมไปถึง พื้นที่ชุ่มน้ำ, ป่า และทุนดรา

ถึงแม้ว่าโดยทั่วไปจะเรียกหญ้าทะเล, กก และวงศ์กกเป็น "หญ้า" แต่ทั้งหมดอยู่นอกวงศ์นี้ โดยกกและวงศ์กกมีความคล้ายคลึงกับหญ้าตรงที่อยู่ในอันดับ Poales แต่หญ้าทะเลอยู่ในอันดับ Alismatales อย่างไรก็ตาม ทั้งหมดอยู่ในกลุ่มพืชใบเลี้ยงเดี่ยว""",
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
    previous_evaluations: list[str] = Field(
        default=[],
        example=["การประเมินที่เคยทำไปก่อนหน้านี้"],
        description="Previous evaluations for context."
    )
    text: str = Field(
        ...,
        example="""หญ้า เป็นวงศ์ของพืชดอกใบเลี้ยงเดี่ยวที่มีจำนวนมากและมีแทบทุกหนแห่ง โดยมีประมาณ 780 สกุลและประมาณ 12,000 สปีชีส์...""",
        description="The original text from which the question was generated."
    )

# Endpoint to generate a question
@app.post("/generate-question/")
async def generate_question(data: QuestionRequest):
    try:
        prompt = generate_question_prompt(data.previous_questions, data.text)
        response = ollama.generate(model="gemma:2b", prompt=prompt, stream=True)
        generated_text = "".join([chunk["response"] for chunk in response])
        
        return {"question": generated_text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating question: {str(e)}")

# Endpoint to evaluate a response
@app.post("/evaluate-response/")
async def evaluate_response(data: EvaluateRequest):
    try:
        prompt = evaluate_response_prompt(data.question, data.user_response, data.previous_evaluations, data.text)
        response = ollama.generate(model="gemma:2b", prompt=prompt, stream=True)
        evaluation_text = "".join([chunk["response"] for chunk in response])
        
        return {"evaluation": evaluation_text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating response: {str(e)}")

# Example FastAPI main application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
