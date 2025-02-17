from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from janus.models import MultiModalityCausalLM, VLChatProcessor
import os
from typing import List, Optional
import uuid
import json
import asyncio
import concurrent.futures
from contextlib import asynccontextmanager
from pydantic import BaseModel
import re  # 导入正则表达式模块

# 模型和处理器 (全局变量，只加载一次)
model_path = r"D:\Code\ML\Model\huggingface\Janus-Pro-7B"
vl_chat_processor = None
vl_gpt = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: Load the model
    await load_model_and_processor()
    yield
    # Shutdown logic:  (Nothing specific needed here for this example)
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源 (开发时可以这样，生产环境要限制)
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法 (GET, POST, PUT, DELETE, OPTIONS, ...)
    allow_headers=["*"],  # 允许所有头部
)



# 定义响应体模型
class CardInfoResponse(BaseModel):
    year: str
    program: str
    card_set: str
    card_num: str
    athlete: str


async def load_model_and_processor():
    global vl_chat_processor, vl_gpt
    print("Loading model and processor in the background...")
    loop = asyncio.get_running_loop()  # 使用get_running_loop
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, _load_model_sync)  # 在线程池中运行同步函数
    print("Model and processor loaded asynchronously.")


def _load_model_sync():
    global vl_chat_processor, vl_gpt

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
    )
    vl_gpt = vl_gpt.eval()
    vl_gpt = torch.compile(vl_gpt)


@app.post("/predict_with_image/", response_model=CardInfoResponse)
async def predict_with_image(image: UploadFile = File(...), question: str = Form(...)):
    empty_result = '''{"year":"","program":"","card_set":"","card_num":"","athlete":""}'''
    try:
        # 保存上传的图片
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        image_path = f"temp/temp_{uuid.uuid4()}.jpg"
        pil_image.save(image_path)
        print(f"Image saved to {image_path}, size: {pil_image.size}")

        # 构建对话 (传入 PIL Image 对象)
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [pil_image],  # 传入 PIL Image, 不是路径
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # 准备模型输入
        prepare_inputs = vl_chat_processor(conversations=conversation, images=[pil_image]).to(
            vl_gpt.device)  # images 传入
        prepare_inputs["pixel_values"] = prepare_inputs["pixel_values"].half()

        # 生成回复
        with torch.no_grad():
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
                bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
                eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
            )

        # 解码回复
        answer = vl_chat_processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(answer)

        # 删除临时图片
        os.remove(image_path)  # 确保删除

        # 使用更健壮的 JSON 解析 (与 predict_text_only 相同)
        try:
            answer_json = json.loads(answer)
        except json.JSONDecodeError as e:
            print('json.JSONDecodeError:', e)
            # 更健壮的 JSON 提取 (尝试多种方法)
            try:
                # 方法 1: 查找 "{...}"
                start = answer.find("{")
                end = answer.rfind("}") + 1
                json_str = answer[start:end]
                answer_json = json.loads(json_str)
            except Exception as e:
                print('json.JSONDecodeError:', e)
                # 方法 2:  (如果模型输出 "输出：{...}" 或类似的格式)
                try:
                    match = re.search(r'\{.*\}', answer)  # 使用正则表达式查找 {}
                    if match:
                        json_str = match.group(0)
                        answer_json = json.loads(json_str)
                    else:
                        answer_json = empty_result
                except Exception as e:
                    print('json.JSONDecodeError:', e)
                    answer_json = empty_result
    except Exception as e:
        print('Exception', e)
        answer_json = empty_result

    return answer_json

text_prompt = """你是一个专业的体育卡牌数据解析助手。请严格按照以下规则, 从文本中提取卡牌信息，输出JSON格式结果：
**绝对不要包含任何其他文本、解释、问候语或对话。只输出 JSON 对象。**

1. 字段说明(输出字段必须为以下名称)：
   - year: 年份(取前四位数字，如'2023-24'取2023)
   - program: 卡系列(例如 Aficionado/Prominence/Honors/Playoffs/Kobe Bryant Box Set/Momentum/Hoops/Leather and Lumber/Court Kings/Contenders Draft Picks /Cooperstown/Gala/Complete/Vertex/Innovation/Luminance/PhotoGenic/Diamond Kings/Prizm)
   - card_set: 卡种(卡系列后的第一个主要特征短语)
   - card_num: 卡编号(以#开头的最早出现的连续字符)
   - athlete: 球员名称(最后出现的人名，需包含姓和名)

2. 处理规则：
   - 字段不存在时设为空字符串
   - 忽略括号内内容和特殊标记(如RC/SP)
   - 保持原始文本顺序，不要重组内容
   - 优先匹配卡系列列表，未匹配到则留空

3. 示例：
Bryan Bresee 2023 Donruss Optic #276 Purple Shock RC New Orleans Saints
{"year":"2023","program":"Donruss Optic","card_set":"Purple Shock","card_num":"276","athlete":"Bryan Bresee"}

"""

@app.post("/predict_text_only/")
async def predict_text_only(question: str = Form(...)):

    empty_result = '''{"year":"","program":"","card_set":"","card_num":"","athlete":""}'''
    try:
        # 构建对话
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{text_prompt}\n{question}",
                "images": [],  # 没有图片
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        if vl_gpt is None:
            return JSONResponse({"error": "Model not loaded yet."}, status_code=503)

        # # 使用 tokenizer 处理纯文本
        encoded_input = vl_chat_processor.tokenizer(conversation[0]["content"], return_tensors="pt").to(vl_gpt.device)
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]

        with torch.no_grad():
            outputs = vl_gpt.language_model.generate(
                input_ids=input_ids,  # 直接传入
                attention_mask=attention_mask,  # 直接传入
                pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
                bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
                eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
                max_new_tokens=128,
                do_sample=False,
            )

        answer = vl_chat_processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(answer)

        # 使用更健壮的 JSON 解析 (与 predict_text_only 相同)
        try:
            answer_json = json.loads(answer)
        except json.JSONDecodeError as e:
            print('json.JSONDecodeError:', e)
            # 更健壮的 JSON 提取 (尝试多种方法)
            try:
                # 方法 1: 查找 "{...}"
                start = answer.find("{")
                end = answer.rfind("}") + 1
                json_str = answer[start:end]
                answer_json = json.loads(json_str)
            except Exception as e:
                print('json.JSONDecodeError:', e)
                # 方法 2:  (如果模型输出 "输出：{...}" 或类似的格式)
                try:
                    match = re.search(r'\{.*\}', answer)  # 使用正则表达式查找 {}
                    if match:
                        json_str = match.group(0)
                        answer_json = json.loads(json_str)
                    else:
                        answer_json = empty_result
                except Exception as e:
                    print('json.JSONDecodeError:', e)
                    answer_json = empty_result
    except Exception as e:
        print('Exception', e)
        answer_json = empty_result

    return answer_json


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=9001)
