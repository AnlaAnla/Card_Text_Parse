
import numpy as np
import os  # 导入 os 模块

# 全局变量 (在 utils.py 中)

def cosine_similarity(vec1, vec2):
    """计算余弦相似度。"""
    vec1 = vec1.reshape(1, -1)
    dot_product = np.dot(vec2, vec1.T)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2, axis=1, keepdims=True)
    similarities = dot_product / (norm_vec1 * norm_vec2)
    return similarities.flatten()


def search_vec2text(model, text, vec_data, name_list, alpha=0.0, top_k=5):
    """
    向量搜索函数。

    Args:
        text: 查询文本。
        vec_data: 向量数据。
        name_list: 名称列表。
        alpha: 长度惩罚系数。
        top_k: 返回最相似结果的数量。

    Returns:
        包含匹配结果的列表，每个结果是一个包含名称和相似度分数的字典。
    """
    output_vec = model.encode(text, normalize_embeddings=True).reshape(1, -1)
    similarities = cosine_similarity(output_vec, vec_data)

    # 长度惩罚
    if alpha > 0.0:  # alpha 应该大于0才进行惩罚
        for i, name in enumerate(name_list):
            similarities[i] *= (1 - alpha * abs(len(text) - len(name)) / max(len(text), len(name)))

    search_results = []
    if len(similarities) < top_k:
        top_k = len(similarities)

    top_k_indices_arr = np.argpartition(similarities, -top_k)[-top_k:]
    top_k_indices_arr = top_k_indices_arr[np.argsort(similarities[top_k_indices_arr])][::-1]

    for i in top_k_indices_arr:
        search_results.append({"name": str(name_list[i]), "similarity": float(similarities[i])})

    return search_results
