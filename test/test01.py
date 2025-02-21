
def sort_tags_by_text_position(text, tag_list):
    """
    根据 tag_list 中的 tag 在 text 中出现的先后顺序进行排序。

    Args:
        text: 要搜索的文本字符串。
        tag_list: 要排序的标签列表。

    Returns:
        排序后的标签列表。
    """

    # 创建一个字典来存储每个 tag 的信息：{tag: (start_index, length)}
    tag_info = {}
    for tag in tag_list:
        start_index = text.find(tag)
        if start_index != -1:  # 只有在 text 中找到 tag 才添加
            tag_info[tag] = (start_index, len(tag))

    # 根据 start_index（出现位置）升序排序，如果 start_index 相同，则根据 length（长度）降序排序
    sorted_tags = sorted(tag_info.keys(), key=lambda tag: (tag_info[tag][0], -tag_info[tag][1]))

    return sorted_tags

text = "Photogenic Tobias Harris Wedges /49"
tag_list = ['PhotoGenic']
print(sort_tags_by_text_position(text, tag_list))