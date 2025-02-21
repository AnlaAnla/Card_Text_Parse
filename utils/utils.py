import re
def preprocess_year_num(text):
    # 尝试匹配 "20xx-yy"、"20xx" 或 "xx-yy" 格式的年份
    year_match = re.search(r'\b(20\d{2})-(\d{2})\b|\b(20\d{2})\b|\b(\d{2})-(\d{2})\b', text)
    num_match = re.search(r'#([A-Za-z0-9\-]+)', text)

    if year_match:
        # "20xx-yy" 格式 (group 1 和 2)
        if year_match.group(1):
            year = year_match.group(1)
        # "20xx" 格式 (group 3)
        elif year_match.group(3):
            year = year_match.group(3)
        # "xx-yy" 格式 (group 4 和 5)
        elif year_match.group(4):
            year = "20" + year_match.group(4)  # 添加 "20" 前缀
            # 简单检查，防止 "99-00" 这样的情况被错误处理
            if int(year) > 2099 or int(year) < 2000:
                year = ""

        else:
            year = ""
    else:
        year = ""

    return {
        'year': year,
        'card_num': num_match.group(1) if num_match else ""
    }


def match_tag(csv_data, tag, text):
    # 和数据库匹配
    matches = csv_data[csv_data[tag].str.contains(f"^{text}$", na=False, regex=True, case=False)]
    return not matches.empty

def filter_dataframe_optimized(dataframe, filter_dict):
    """
    根据给定的字典中的字段筛选 DataFrame

    Args:
        dataframe: 要筛选的 Pandas DataFrame。
        filter_dict: 包含筛选条件的字典。键是 DataFrame 的列名，值是筛选条件。
                     支持的键: 'program_new', 'card_num', 'athlete_new'
                     如果值为 None 或空字符串，则忽略该筛选条件。
                     如果键是 'card_num' 且值不是纯数字字符串，则忽略。

    Returns:
        筛选后的 DataFrame。
    """

    mask = True  # 初始 mask 为 True

    for column, value in filter_dict.items():
        if not value:  # 等价于 if value is None or value == "":
            continue

        if column == 'card_num':
            if not isinstance(value, str) or not value.isdigit():
                continue
            value = str(value)  # card_num 转为字符串
        elif column not in ('program_new', 'athlete_new'):  # 优化点1
            continue

        if column in dataframe.columns:  # 优化点2
            mask = mask & (dataframe[column] == value)

    return dataframe[mask]


def sort_tags_by_text_position(text:str, tag_list:list[str]):
    """
    根据 tag_list 中的 tag 在 text 中出现的先后顺序进行排序。

    Args:
        text: 要搜索的文本字符串。
        tag_list: 要排序的标签列表。

    Returns:
        排序后的标签列表。
    """
    text = text.lower().strip()

    # 创建一个字典来存储每个 tag 的信息：{tag: (start_index, length)}
    tag_info = {}
    for tag in tag_list:
        start_index = text.find(tag.lower().strip())
        if start_index != -1:  # 只有在 text 中找到 tag 才添加
            tag_info[tag] = (start_index, len(tag.split(' ')))

    # 根据 start_index（出现位置）升序排序，如果 start_index 相同，则根据 length（长度）降序排序
    sorted_tags = sorted(tag_info.keys(), key=lambda tag: (tag_info[tag][0], -tag_info[tag][1]))

    return sorted_tags