from typing import List, Any

import pandas as pd
import os


def csv2tag_list(data, tag):
    athlete_new_list: list[str] = list(data[tag].dropna().unique())
    for i in range(len(athlete_new_list)):
        athlete_new_list[i] = athlete_new_list[i].strip()

    print(tag, " list: ", len(athlete_new_list))
    athlete_new_set = set(athlete_new_list)
    print(tag, " set: ", len(athlete_new_set))
    athlete_new_list = sorted(list(athlete_new_set))
    return athlete_new_list


def csv2tag_txt(data_list, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for text in data_list:
            f.write(text + '\n')
    print('end')


if __name__ == '__main__':

    filename = r"D:\Code\ML\Text\card\checklist_2023.csv"
    data = pd.read_csv(filename, low_memory=False)

    tag_list = ['program_new', 'card_set', 'athlete_new']

    for tag in tag_list:
        save_path = os.path.join(r'D:\Code\ML\Text\checklist_tags\2023', tag + '.txt')
        data_list = csv2tag_list(data, tag)
        csv2tag_txt(data_list, save_path)
        print(save_path)
        print('_'*20)
