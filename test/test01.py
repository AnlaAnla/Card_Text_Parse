import re

program_names = ("Aficionado/Prominence/Honors/Playoffs/Kobe Bryant Box Set/"
                     "Momentum/Hoops/Leather and Lumber/Court Kings/Contenders Draft Picks /"
                     "Cooperstown/Gala/Complete/Vertex/Innovation/Luminance/PhotoGenic/Diamond Kings/Prizm")
json_formate = '''{"year":"2023","program":"Donruss Optic","card_set":"Purple Shock","card_num":"276","athlete":"Bryan Bresee"}'''

# 系统提示词 (图片和文本)
image_text_prompt = f"""你是一个专业的体育卡牌数据解析助手。请严格按照以下规则, 从图片和文本中提取卡牌信息，输出JSON格式结果：
**绝对不要包含任何其他文本、解释、问候语或对话。只输出 JSON 对象。**

1. 字段说明(输出字段必须为以下名称)：
   - year: 年份(取前四位数字，如'2023-24'取2023)
   - program: 卡系列(例如 {program_names})
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
已知信息:
- 年份：2023
- 卡号：276
Bryan Bresee 2023 Donruss Optic #276 Purple Shock RC New Orleans Saints
{json_formate}
"""

def preprocess_year_num(text):
    year_match = re.search(r'\b(20\d{2})(?:-\d{2})?\b', text)
    num_match = re.search(r'#([A-Za-z0-9\-]+)', text)
    return {
        'year': year_match.group(1) if year_match else "",
        'card_num': num_match.group(1) if num_match else ""
    }


text = "CHASE YOUNG - 2020 Prizm Base Rookie - Mint PSA 9 - plus bonus"
preprocess_year_num_result = preprocess_year_num(text)

image_text_prompt = f'''
{image_text_prompt}
已知信息:
- 年份：{preprocess_year_num_result['year']}
- 卡号：{preprocess_year_num_result['card_num']}
{text}
'''

print(image_text_prompt)