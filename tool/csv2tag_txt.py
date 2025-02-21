import pandas as pd
import os

check_list_dir = r"D:\Code\ML\Text\card"

program_list = []
for filename in os.listdir(check_list_dir):
    file_path = os.path.join(check_list_dir, filename)
    data = pd.read_csv(file_path, low_memory=False)
    data_list = list(data['team'].dropna().unique())

    # for item in data_list:
    #     if 'Diego Lugano' in item:
    #         print(filename)

    # print(f"{filename}: {len(data_list)}")
    # program_list += data_list

# print('merge data: ', len(program_list))
# new_program_list = []
#
# for name in program_list:
#     if '|' in name or '/' in name:
#         temp_list = []
#         if '|' in name:
#             temp_list += name.split('|')
#         elif '/' in name:
#             temp_list += name.split('/')
#
#         for i in range(len(temp_list)):
#             temp_list[i] = temp_list[i].strip()
#
#         new_program_list += temp_list
#     else:
#         new_program_list.append(name.strip())
#
# print("merge2 data: ", len(new_program_list))
#
#
# new_program_list = set(new_program_list)
# print('merge set: ', len(new_program_list))
# new_program_list = sorted(list(new_program_list))
#
# with open(r'D:\Code\ML\Text\checklist_tags\team_tags.txt', 'w', encoding='utf-8') as f:
#     for text in new_program_list:
#         f.write(text + '\n')
# print('end')
