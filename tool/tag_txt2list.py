def get_data_list(data_path):
    data_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            tag = line.strip()
            if tag:  # 排除空行
                data_list.append(tag)
    print('length: ', len(data_list))
    return data_list

if __name__ == '__main__':

    name_path = r"D:\Code\ML\Text\checklist_tags\athlete_tags.txt"
    team_path = r"D:\Code\ML\Text\checklist_tags\team_tags.txt"

    name_list = get_data_list(name_path)
    team_list = get_data_list(team_path)

    # new_name_list = sorted(list(set(name_list) - set(team_list)))
    name_set = set(name_list)
    team_set = set(team_list)
    print(team_set & name_set)
    print()

    # with open(r'D:\Code\ML\Text\checklist_tags\athlete_tags2.txt', 'w', encoding='utf-8') as f:
    #     for text in new_name_list:
    #         f.write(text + '\n')
    # print('end')