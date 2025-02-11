from fastapi import FastAPI, HTTPException
from model import SearchResponse, SearchRequest
from utils import search_vec2text
from sentence_transformers import SentenceTransformer
import numpy as np
import os

app = FastAPI()

model = SentenceTransformer('BAAI/bge-large-en-v1.5')  # 或其他模型
program_vec_data = None
program_name_list = None
cardSet_vec_data = None
cardSet_name_list = None


def load_data():
    """加载向量和名称数据。"""
    global program_vec_data, program_name_list, cardSet_vec_data, cardSet_name_list

    # 使用 os.path.join 构建文件路径
    program_vec_path = os.path.join("Data", "program_vec.npy")
    program_names_path = os.path.join("Data", "program_vec_names.npy")
    cardSet_vec_path = os.path.join("Data", "cardSet_vec.npy")
    cardSet_names_path = os.path.join("Data", "cardSet_vec_names.npy")

    program_vec_data = np.load(program_vec_path)
    program_name_list = np.load(program_names_path)
    cardSet_vec_data = np.load(cardSet_vec_path)
    cardSet_name_list = np.load(cardSet_names_path)
    print("已加载向量和名称数据")


# 在应用启动时加载数据
@app.on_event("startup")
async def startup_event():
    load_data()


@app.post("/search_program", response_model=SearchResponse)
async def search_program(request: SearchRequest):
    """
    搜索 program。
    """
    if program_vec_data is None or program_name_list is None:
        raise HTTPException(status_code=503, detail="数据未加载完成")
    results = search_vec2text(model, request.text, program_vec_data, program_name_list, top_k=request.topk)
    return {"results": results}


@app.post("/search_cardSet", response_model=SearchResponse)
async def search_cardSet(request: SearchRequest):
    """
    搜索 card_set。
    """
    if cardSet_vec_data is None or cardSet_name_list is None:
        raise HTTPException(status_code=503, detail="数据未加载完成")

    results = search_vec2text(model, request.text, cardSet_vec_data, cardSet_name_list, top_k=request.topk)
    return {"results": results}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=9002)
