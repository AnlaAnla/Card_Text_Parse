import requests
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

import pandas as pd
import re
from utils.call_predict_with_image import call_predict_with_image
from utils.call_predict_with_image2 import call_predict_with_image2
from utils.program_cardSet_vecSearch import text_vecSearch
from utils.ebay_text_image_parse import ebay_text_image_parse

app = FastAPI()
request_num = 0
# 跨域解决
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源 (开发时可以这样，生产环境要限制)
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法 (GET, POST, PUT, DELETE, OPTIONS, ...)
    allow_headers=["*"],  # 允许所有头部
)


class InputData(BaseModel):
    ebay_text: str
    # image_url: str


@app.post("/parse_ebay_data/")
async def parse_ebay_data(input_data: InputData):
    """
    接收 ebay_text, 返回json字段
    样例
    ebay_text = "2021-22 Panini Prizm Red Ice Prizm Tim Duncan #268 HOF"
    """
    global request_num
    request_num += 1
    try:
        print(
            f'''
    ======================================
    | 接收的请求数量: [{request_num}]        |
    ======================================
            '''
              )
        print('||||---- input data: ', input_data)

        llm_output = ebay_text_image_parse(input_data.ebay_text, "Data/temp.jpg")

        print('||||---- return data: ', llm_output)
        return llm_output
    except Exception as e:
        # 更好的错误处理，可以记录更详细的错误信息
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


if __name__ == "__main__":
    # ebay_text = "2021-22 Panini Prizm Red Ice Prizm Tim Duncan #268 HOF"
    # image_url = "https://i.ebayimg.com/images/g/WdUAAOSw2jhmy8s7/s-l1200.jpg"

    # LLM_output = ebay_text_image_parse(ebay_text=ebay_text, image_url=image_url)

    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=9000)
