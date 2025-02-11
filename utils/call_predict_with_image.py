import requests
import json
from PIL import Image
import io

# 系统提示词 (图片和文本)
image_text_prompt = """你是一个专业的体育卡牌数据解析助手。请严格按照以下规则, 从图片和文本中提取卡牌信息，输出JSON格式结果：
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
   - **优先从图片中提取各个字段信息, 如果图片中信息不全, 则从文本中提取**

3. 示例：
Bryan Bresee 2023 Donruss Optic #276 Purple Shock RC New Orleans Saints
{"year":"2023","program":"Donruss Optic","card_set":"Purple Shock","card_num":"276","athlete":"Bryan Bresee"}
"""


def call_predict_with_image(url, image_path: str, question: str):
    """
    Args:
        image_path: 图片路径（可以是本地路径或网络 URL）。
        question:  要提出的问题（关于卡牌的）。

    Returns:
        一个字典，包含 API 的响应。
    """

    prompt_with_question = f"{image_text_prompt}\n{question}"
    # print(prompt_with_question)

    # 检查 image_path 是本地路径还是 URL
    if image_path.startswith("http://") or image_path.startswith("https://"):
        # 如果是 URL，直接读取
        try:
            response = requests.get(image_path, stream=True)
            response.raise_for_status()  # 检查请求是否成功
            image = Image.open(io.BytesIO(response.content))
            # 将 PIL Image 对象转换为 bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')  # 或其他格式
            img_byte_arr = img_byte_arr.getvalue()
            files = {"image": ("image.jpg", img_byte_arr, "image/jpeg")}

        except requests.exceptions.RequestException as e:
            print(f"Error downloading image from URL: {e}")
            return {"error": f"Error downloading image from URL: {e}"}
    else:
        # 如果是本地路径，打开文件
        try:
            files = {"image": open(image_path, "rb")}
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return {"error": f"Image file not found at {image_path}"}

    data = {"question": prompt_with_question}

    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()  # 检查请求是否成功 (状态码 2xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return {"error": f"Error calling API: {e}"}
    finally:
        if 'files' in locals() and isinstance(files["image"], io.IOBase):  # 关闭文件
            files["image"].close()
