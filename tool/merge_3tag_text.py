import pandas as pd
import os


def read_txt_file(file_path):
    """读取 txt 文件，返回一个包含非空行的列表。"""
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tag = line.strip()
            if tag:  # 排除空行
                data_list.append(tag)
    return data_list


def create_combined_csv(program_file, card_set_file, athlete_file, output_csv):
    """
    将三个 txt 文件合并成一个 CSV 文件。

    Args:
        program_file: program 文件的路径。
        card_set_file: card_set 文件的路径。
        athlete_file: athlete 文件的路径。
        output_csv: 输出 CSV 文件的路径。
    """

    # 读取三个 txt 文件
    program_list = read_txt_file(program_file)
    card_set_list = read_txt_file(card_set_file)
    athlete_list = read_txt_file(athlete_file)

    # 找到最长的列表长度
    max_len = max(len(program_list), len(card_set_list), len(athlete_list))

    # 使用 None 填充较短的列表，使它们的长度相同
    program_list.extend([None] * (max_len - len(program_list)))
    card_set_list.extend([None] * (max_len - len(card_set_list)))
    athlete_list.extend([None] * (max_len - len(athlete_list)))

    # 创建 DataFrame
    df = pd.DataFrame({
        'program': program_list,
        'card_set': card_set_list,
        'athlete': athlete_list
    })

    # 保存为 CSV 文件
    df.to_csv(output_csv, index=False)  # index=False 表示不保存行索引
    print(f"已创建 CSV 文件: {output_csv}")


if __name__ == '__main__':
    save_dir = "../Data"
    # 假设你的文件都在同一个目录下
    base_dir = r"D:\Code\ML\Text\checklist_tags"  # 你的目录
    program_file = os.path.join(base_dir, "program_tags.txt")
    card_set_file = os.path.join(base_dir, "cardSet_tags.txt")
    athlete_file = os.path.join(base_dir, "athlete_tags.txt")
    output_csv = os.path.join(save_dir, "program_cardSet_athlete.csv")

    create_combined_csv(program_file, card_set_file, athlete_file, output_csv)
