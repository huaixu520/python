# -*- coding: utf-8 -*-
"""文本处理后，对分析好的数据可视化，生成词云和词频饼图"""
from wordcloud import WordCloud  # 词云显示模块
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import mpl  # 饼状图默认无中文显示，需显示中文则引入模块
from PIL import Image  # 图像处理,加载词云图片模板
import webbrowser  # 用于打开浏览器
import tkinter as tk  # 窗口视窗设计,GUI界面
import tkinter.messagebox  # 窗口视窗设计,弹窗
from tkinter import END  # 末尾显是输出
from tkinter import scrolledtext  # 滚动文本框


def all_world_list(file_path):  # file_path : 名称的路径
    all_list = [line1.strip() for line1 in open(file_path, 'r', encoding='utf-8').readlines()]  # 清除空格
    return all_list


# 打开处理后文本
text_cloud = open('./dataset/wordCount.txt', 'r', encoding='UTF-8').read()
data = pd.read_excel(r'./dataset/wordCount.xls')  # 加载词汇表格文件

# 统计两地出现次数：南方和北方
two_places_list = []  # 两地及其出现次数保存
places = all_world_list('./dataset/base_data/two_places.txt')  # 这里加载两地名称的路径

# 统计三天出现次数：昨天、今天、明天
three_days_list = []  # 三天及其出现次数保存
days = all_world_list('./dataset/base_data/three_days.txt')  # 这里加载三天名称的路径

# 统计四季出现次数：春夏秋冬
four_seasons_list = []  # 四季及其出现次数保存
seasons = all_world_list('./dataset/base_data/four_seasons.txt')  # 这里加载四季名称的路径

n = 0  # 计数，计算两地出现的下标
for word_place in data['world']:
    if word_place not in places:
        n += 1
    else:
        place = data.at[n, 'count']  # 若当前分词在对比词库中，则查找当前位置count值，
        two_places_list.append(word_place)
        two_places_list.append(place)  # 将当前分词和其对应count值追加在
p = np.array(two_places_list).reshape((2, 2))  # 生成数组，并改变数组形状，使一列为word，一列为count
place_word_list = p[:, 0]  # 选择第一列
place_count_list = p[:, 1]  # 选择第二列

# 统计三天出现次数：昨天、今天、明天
n = 0  # 重置
for word_day in data['world']:
    if word_day not in days:
        n += 1
    else:
        day = data.at[n, 'count']  # 若当前分词在对比词库中，则查找当前位置count值，
        three_days_list.append(word_day)
        three_days_list.append(day)  # 将当前分词和其对应count值追加在
d = np.array(three_days_list).reshape((3, 2))  # 生成数组，并改变数组形状，使一列为word，一列为count
day_word_list = d[:, 0]  # 选择第一列
day_count_list = d[:, 1]  # 选择第二列

# 统计四季出现次数：春夏秋冬
n = 0  # 重置
for word_season in data['world']:
    if word_season not in seasons:
        n += 1
    else:
        season = data.at[n, 'count']  # 若当前分词在对比词库中，则查找当前位置count值，
        four_seasons_list.append(word_season)
        four_seasons_list.append(season)  # 将当前分词和其对应count值追加在
s = np.array(four_seasons_list).reshape((4, 2))  # 生成数组，并改变数组形状，使一列为word，一列为count
season_word_list = s[:, 0]  # 选择第一列
season_count_list = s[:, 1]  # 选择第二列

# 生成词云对象
# 用matplotlib导入图片出现 UserWarning: mask image should be unsigned byte between 0 and 255警告
# 参照https://blog.csdn.net/yuntunlu/article/details/105226886
word_cloud_shape = np.array(Image.open("./dataset/base_data/guitar.png"))  # 加载词云形状图片
wc = WordCloud(font_path="C:/Windows/Fonts/msyhbd.ttc",  # 显示中文，从属性里复制字体名称，不能直接看windows显示的字体名
               width=1000,  # 图片的宽
               height=860,  # 图片的长
               max_words=200,  # 最大词数
               background_color='white',  # 背景色设置为白色
               mask=word_cloud_shape,  # 改变词云形状图片
               scale=5  # 按照比例进行放大画布，如设置为5，则长和宽都是原来画布的5倍，使图像更清晰。
               ).generate(text_cloud)

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 饼状图黑体


def word_cloud():
    # 创建词云画布
    plt.figure('word_cloud', figsize=(8, 5))  # 画布重命名，否则不能显示多张画布,然后调节画布大小为8:5
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')  # 不显示X轴和Y轴
    plt.show()  # 显示画布图片
    # 保存词云到文件
    wc.to_file('./dataset/wordcloud.png')  # wordcloud库提供保存函数 ：to_file()
    show_text.delete(1.0, "end")  # 从第一行清除到最后一行,防止重复显示
    # 分析显示
    show_text.insert(END, '结论：世界、生活、时间字体较大、故出现频率较高，可见民谣歌手感叹韶华易逝，青春小鸟一去不回来。他们会觉得很孤单，'
                          '但是并不沉浸在忧伤之中，而是心中向往着远方，对整个世界充满希望、对未来充满阳光。\n')


def place_count():
    # 创建饼图和柱状图画布
    plt.figure('two_places', figsize=(8, 5))  # 画布重命名，否则不能显示多张画布: 两方,然后调节画布大小为8:5
    plt.subplot(121)  # 将整个figure分成一行两列，并该图形放在第1个网格
    plt.title("两方数量柱状图")
    # 绘图 x x轴， height 高度, 默认：color="blue", width=0.8
    plt.bar(x=place_word_list, height=list(map(int, list(place_count_list))), width=0.5)
    plt.subplot(122)  # 将整个figure分成一行两列，并该图形放在第2个网格
    plt.title("两方比例饼图图")
    plt.pie(x=place_count_list, labels=place_word_list, autopct='%0.2f%%')
    plt.savefig('./dataset/two_places.png')  # matplotlib 提供 plt.savefig()，保存饼图
    plt.show()  # 显示画布图片
    show_text.delete(1.0, "end")  # 从第一行清除到最后一行,防止重复显示
    # 分析显示
    show_text.insert(END, '结论：歌手们更多的念叨着南方而不是北方，南方比北方多了11.12%。虽然平时听到的城市很多在北方，特别是北京，'
                          '但是歌手们心心念念的还是南方这种气候温和的地方。\n')


def day_count():
    plt.figure('three_days', figsize=(8, 5))  # 画布重命名，否则不能显示多张画布: 三天,然后调节画布大小为8:5
    plt.subplot(121)  # 将整个figure分成一行两列，并该图形放在第1个网格
    plt.title("三天数量柱状图")
    # 绘图 x x轴， height 高度, 默认：color="blue", width=0.8
    plt.bar(x=day_word_list, height=list(map(int, list(day_count_list))), width=0.5)
    plt.subplot(122)  # 将整个figure分成一行两列，并该图形放在第2个网格
    plt.title("三天比例饼图图")
    plt.pie(x=day_count_list, labels=day_word_list, autopct='%0.2f%%')
    plt.savefig('./dataset/three_days.png')  # matplotlib 提供 plt.savefig()，保存饼图
    plt.show()  # 显示画布图片
    show_text.delete(1.0, "end")  # 从第一行清除到最后一行,防止重复显示
    # 分析显示
    show_text.insert(END, '结论：看得出，民谣歌手们是在向前看，是往未来寄托希望，仅少部分是缅怀过去和活在当下的，'
                          '明天这个词在歌词中出现的次数最多，接着是今天和昨天。\n')


def season_count():
    plt.figure('four_seasons', figsize=(8, 5))  # 画布重命名，否则不能显示多张画布: 四季,然后调节画布大小为8:5
    plt.subplot(121)  # 将整个figure分成一行两列，并该图形放在第1个网格
    plt.title("四季数量柱状图")
    # 绘图 x x轴， height 高度, 默认：color="blue", width=0.8
    plt.bar(x=season_word_list, height=list(map(int, list(season_count_list))), width=0.5)
    plt.subplot(122)  # 将整个figure分成一行两列，并该图形放在第2个网格
    plt.title("四季比例饼图图")
    plt.pie(x=season_count_list, labels=season_word_list, autopct='%0.2f%%')
    plt.savefig('./dataset/four_seasons.png')  # matplotlib 提供 plt.savefig()，保存饼图
    plt.show()  # 显示画布图片
    show_text.delete(1.0, "end")  # 从第一行清除到最后一行,防止重复显示
    # 分析显示
    show_text.insert(END, '结论：由饼图可以看出，民谣歌手们比较喜欢春天~~等待下一个春天回来；飘在异乡的雪覆盖了春天。\n')


# 创建一个名字为win的主窗口对象
win = tk.Tk()
win.title('槐序')
win.resizable(0, 0)  # 禁止调整窗口大小
win.geometry('500x380')  # 设置窗口大小
tk.Label(win, text='BY:', font=('SimHei', 20)).place(x=20, y=10)
tk.Label(win, text='智能一向洁', font=('SimHei', 20)).place(x=50, y=40)


def introduce():
    """定义一个函数，用于输出程序介绍"""
    show_text.insert(END, '我分析了45万字的歌词，为了搞清楚民谣歌手们在唱些什么\n')
    return None


def clear_output():
    """定义一个函数，用于清空输出框的内容"""
    show_text.delete(1.0, "end")  # 从第一行清除到最后一行


def contact_me():
    """定义一个函数，用于打开第三方网站"""
    webbrowser.open("https://user.qzone.qq.com/2268289662")
    return None


def quited():
    """定义一个函数，用于退出程序"""
    exit_button = tk.messagebox.askokcancel('确认', '确定退出吗？')
    if exit_button:
        win.destroy()
    return None


# 设置程序介绍按钮
introduce_button = tk.Button(win, bg='skyblue', text='程序介绍', font=('SimHei', 10), width=8, height=2, command=introduce)
introduce_button.place(x=40, y=80)

# 设置清空输出按钮
clear_button = tk.Button(win, bg='skyblue', text='清空输出', font=('SimHei', 10), width=8, height=2, command=clear_output)
clear_button.place(x=220, y=80)
# 设置联系作者按钮
# contact_button = tk.Button(win, text='联系我', font=('SimHei', 10), width=8, height=1, command=contact_me)
# contact_button.place(x=220, y=350)
# 设置退出程序按钮
quit__button = tk.Button(win, bg='skyblue', text='关闭程序', font=('SimHei', 10), width=8, height=2, command=quited)
quit__button.place(x=400, y=80)
#  设置词云按钮
word_cloud_button = tk.Button(win, bg='skyblue', text='词云显示', font=('SimHei', 10), width=8, height=2, command=word_cloud)
word_cloud_button.place(x=40, y=130)
# 设置两方（南方、北方）画布按钮
word_cloud_button = tk.Button(win, bg='skyblue', text='两方饼图', font=('SimHei', 10), width=8, height=2, command=place_count)
word_cloud_button.place(x=160, y=130)
# 设置三天（昨天、今天、明天）画布按钮
word_cloud_button = tk.Button(win, bg='skyblue', text='三天饼图', font=('SimHei', 10), width=8, height=2, command=day_count)
word_cloud_button.place(x=280, y=130)
# 设置四季（春夏秋冬）画布按钮
word_cloud_button = tk.Button(win, bg='skyblue', text='四季饼图', font=('SimHei', 10), width=8, height=2, command=season_count)
word_cloud_button.place(x=400, y=130)
# 设置带滑轮显示框
show_text = scrolledtext.ScrolledText(win, width=60, height=10, font=('SimHei', 10))
show_text.place(x=40, y=210)
# 调用组件的mainloop()方法，进入事件循环
win.mainloop()
