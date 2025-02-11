import requests
import json
from PIL import Image
import io


# 系统提示词 (图片和文本)
def create_second_stage_prompt(ebay_text, year, card_num, athlete, top_k_programs, top_k_card_sets):
    """
    创建第二阶段的 LLM 提示词。

    Args:
        ebay_text: 原始 eBay 文本。
        year: 提取出的年份。
        card_num: 提取出的卡号。
        athlete: 提取出的球员姓名。
        top_k_programs: 向量搜索得到的 top-K program 列表。
        top_k_card_sets: 向量搜索得到的 top-K card_set 列表。

    Returns:
        LLM 提示词字符串。
    """

    program_options = ", ".join(top_k_programs)
    card_set_options = ", ".join(top_k_card_sets)

    prompt = f"""你是一个专业的体育卡牌数据解析助手。请根据以下信息，从 eBay 文本中提取卡牌的 program 和 card_set 字段，并以 JSON 格式输出结果：

**绝对不要包含任何其他文本、解释、问候语或对话。只输出 JSON 对象。**

要求：
- 输出结果必须是严格的 JSON 格式。
- program 和 card_set 字段的值必须来自提供的选项（或为空）。
- program 和 card_set 需要严谨选择, 如果差异很多, 尽量设置为空, 设置为空也不要出错

示例：

已知信息：
- 年份:2021
- 卡号:23
- 球员姓名:Lebron James
eBay 文本：Panini Prizm Mosaic  Blue Reactive
可能的 Program 选项：Panini Prizm, Panini Mosaic, Panini Prizm Mosaic
可能的 Card Set 选项：Blue, Reactive, Blue Reactive
输出:
{{"year":"2021", "program": "Panini Prizm Mosaic", "card_set": "Blue Reactive", "card_num":"23","athlete":"Lebron James"}}

已知信息：
- 年份:{year}
- 卡号:{card_num}
- 球员姓名:{athlete}

eBay 文本 (已去除年份、卡号和球员姓名):
{ebay_text}

可能的 Program 选项 (来自向量搜索):
{program_options}

可能的 Card Set 选项 (来自向量搜索):
{card_set_options}
输出:
"""

    return prompt


def call_predict_with_image2(url, image_path: str, ebay_text: str, year, card_num, athlete, top_k_programs, top_k_card_sets):
    """

    Args:
        image_path: 图片路径（可以是本地路径或网络 URL）。
        question:  要提出的问题（关于卡牌的）。

    Returns:
        一个字典，包含 API 的响应。
    """

    prompt_with_question = create_second_stage_prompt(ebay_text=ebay_text,
                                                      year=year,
                                                      card_num=card_num,
                                                      athlete=athlete,
                                                      top_k_programs=top_k_programs,
                                                      top_k_card_sets=top_k_card_sets)
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
