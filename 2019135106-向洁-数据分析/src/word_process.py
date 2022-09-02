# -*- coding:utf-8 -*-
"""用结巴分词进行歌词清理，统计词频等操作"""
import re  # 正则 ，除去标点
import jieba  # 结巴库，对歌词清理，分词
import jieba.analyse
import xlwt  # 写入excel操作


def stopwords_list(file_path):  # 加载停用词
    stopwords = [line1.strip() for line1 in open(file_path, 'r', encoding='utf-8').readlines()]
    return stopwords


# 对句子进行分词
def lyric_count(sentence):
    lyrics_sentences = jieba.cut(sentence.strip())  # 用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
    stopwords = stopwords_list('../dataset/base_data/stopwords.txt')  # 这里加载停用词的路径
    out_str = ''
    for word in lyrics_sentences:
        if word not in stopwords:
            if word != '\t':
                out_str += word
                out_str += " "
    return out_str


def lyric_process():
    # 分词统计
    # 把分词后的结果按`词语+频数`的格式保存为txt和excel文件，方便Tableau处理加工
    print("开始处理歌词······")
    inputs = open('../dataset/base_data/result.txt', 'r', encoding='utf-8')  # 源文件是'utf-8'编码，
    outputs = open('../dataset/all_output_lyric.txt', 'w', encoding='utf-8')  # 保存为utf-8编码
    for line in inputs:
        line_sentence = lyric_count(line)  # 这里的返回值是字符串
        # 注意有些符号是中英文两种格式，所以都要包含进去
        line_sentence = re.sub(
            "[A-Za-z0-9\[\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\'\,\[\]\.\<\>\/\?\~\！\@\#\\\&\*\%\´\：\（\）]",
            "", line_sentence)  # 正则替换符号为空格
        outputs.write(line_sentence)
        outputs.write('\n')
    outputs.close()
    inputs.close()
    wbk = xlwt.Workbook(encoding='utf-8')
    sheet = wbk.add_sheet("wordCount")  # Excel单元格名字
    word_list = []
    key_list = []
    for line in open('../dataset/all_output_lyric.txt', 'rb'):  # 需要分词统计的文档
        item = line.strip('\n\r'.encode('utf-8')).split('\t'.encode('utf-8'))  # 制表格切分
        tags = jieba.analyse.extract_tags(item[0])  # jieba分词
        for t in tags:
            word_list.append(t)
    word_dict = {}
    with open("../dataset/wordCount.txt", 'bw') as wf2:  # 打开文件
        for item in word_list:
            if item not in word_dict:  # 统计数量
                word_dict[item] = 1
            else:
                word_dict[item] += 1
        orderList = list(word_dict.values())
        orderList.sort(reverse=True)
        for i in range(len(orderList)):
            for key in word_dict:
                if word_dict[key] == orderList[i]:
                    wf2.write((key + ' ' + str(word_dict[key])).encode('utf-8'))  # 写入txt文档
                    wf2.write('\n'.encode('utf-8'))  # 写入txt文档
                    key_list.append(key)
                    word_dict[key] = 0
    sheet.write(0, 0, label='world')  # 设置列名，方便后面指定获取关键词
    sheet.write(0, 1, label='count')
    for i in range(1, 250):  # 为了便于统计，限定250个词
        sheet.write(i, 1, label=orderList[i - 1])
        sheet.write(i, 0, label=key_list[i - 1])
    wbk.save('../dataset/wordCount.xls')  # 保存为 wordCount.xls文件
    print("歌词处理完毕······")

if __name__ == "__main__":
    lyric_process()
