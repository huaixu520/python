"""合并所有歌词"""

import os
import time

from word_process import lyric_process  # 引入歌词处理模块


def merge_lyrics():
    def get_list(path):  # 筛选指定目录下所有后缀为.txt的指定文件
        path_1 = os.listdir(path)
        pat = [os.path.join(path, f) for f in path_1 if f.endswith('.txt')]
        return pat

    print("下载完毕！！开始合并歌词······")
    time.sleep(1)  # 用户缓冲时间
    path_0 = '../dataset/lyric'  # 指定筛选文件夹
    all_suffix_is_txt_path = get_list(path_0)  # 获取所有后缀为.txt文件的路径

    # 打开现有文件或创建文件，并清空文件内容，防止多次运行后歌词重复
    with open('../dataset/base_data/result.txt', 'w', encoding='utf-8') as pf:
        pf.close()

    # 将所有歌词文件写入result.txt中
    for i in all_suffix_is_txt_path:
        with open(i, 'r', encoding='utf-8') as pf1:  # 以只读形式将每个歌词文件读出来
            re = pf1.read()
            pf1.close()
        with open('../dataset/base_data/result.txt', 'a', encoding='utf-8') as pf2:  # 以追加方式写入文件到result
            pf2.write(re)
            pf2.close()
    print('合并完成！！')
    time.sleep(1)
    lyric_process()  # 调用word_process下的lyric_process函数，处理歌词


if __name__ == '__main__':
    merge_lyrics()
