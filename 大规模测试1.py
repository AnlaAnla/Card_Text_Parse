import pandas as pd
from utils.ebay_text_image_parse import ebay_text_image_parse_LLM, ebay_text_image_parse
import time


if __name__ == "__main__":

    # question_with_image = "2021 Panini Chronicles Travis Etienne Jr. RC Clear Vision Acetate Rookie #CVR-15"
    # image_url = "https://i.ebayimg.com/images/g/JCQAAOSwHcdlPEGe/s-l1200.jpg"
    test_data_csv_path = r"D:\Code\ML\Text\test\test2023年200个.xlsx"
    save_path = r"D:\Code\ML\Text\test\test2023年200个_第1次测试.xlsx"
    test_data_csv = pd.read_excel(test_data_csv_path)

    for i in range(0, len(test_data_csv)):
    # for i in range(302):
        t1 = time.time()
        print("="*20)
        print('第 ', i)

        # img_url = "https:" + test_data_csv['img'][i]
        # 不用图片了
        # image_url = "Data/temp.jpg"
        ebay_text = test_data_csv['name'][i]

        # output = ebay_text_image_parse_LLM(ebay_text, image_url=image_url)
        output = ebay_text_image_parse(ebay_text)

        test_data_csv.at[i, 'year'] = output['year']
        test_data_csv.at[i, 'program'] = output['program']
        test_data_csv.at[i, 'card_set'] = output['card_set']
        test_data_csv.at[i, 'card_num'] = output['card_num']
        test_data_csv.at[i, 'athlete'] = output['athlete']

        if i % 25 == 0:
            test_data_csv.to_excel('temp.xlsx', index=False)
            print("保存temp.xlsx")

        print("=" * 20, ' |time: ', time.time() - t1)

    test_data_csv.to_excel(save_path, index=False)
    print('end')


