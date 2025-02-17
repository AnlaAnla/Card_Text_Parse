def judge_by_vec_search_list(ebay_text: str, vector_list: list[str], pass_word_list: list[str] = None):
    # 用向量搜索的结果对比原文本
    '''
    :param ebay_text:
    :param vector_list:
    :param pass_word_list: 搜索文本中忽略这些单词的匹配
    :return:
    '''
    # 根据单词数量从多到少排序
    vector_list = sorted(vector_list, key=lambda s: len(s.split()), reverse=True)

    # # 全部小写
    for i in range(len(pass_word_list)):
        pass_word_list[i] = pass_word_list[i].lower()

    for program in vector_list:
        set_program_flat = False
        for word in program.split(' '):
            if word.lower() in pass_word_list or word == '':
                continue

            if word.lower() in ebay_text.lower().split(' '):
                set_program_flat = True
            else:
                set_program_flat = False
                break
        if set_program_flat:
            return program
    return False


vec_list = ['Base Red Cracked Ice Prizm', 'Base Red Ice Prizm', 'Base Prizm Red Ice', 'Brilliance Prizms Red Ice', 'Base Prizms Red Ice', 'All American Prizms Red Ice', 'Fearless Prizms Red Ice', 'Base Hoops Tribute Red Cracked Ice Prizm', 'Flashback Autographs Prizms Red Ice', 'Rookies Prizm Red Ice']

print(judge_by_vec_search_list(ebay_text="2021-22 Panini Prizm Red Ice Prizm Tim Duncan #268 HOF",
                               vector_list=vec_list,
                               pass_word_list=['base']))