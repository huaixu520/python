# -*- coding: utf-8 -*-
"""爬虫主函数，通过歌词id下载歌词"""
import requests
import json
import re
import random
import time
import os

from user_agents import agents  # 从user_agents引入agents，头部
from music_ids import music_id_set, music_name_set  # 从ids中引入歌曲id，歌曲名
from merge_all_lyric import merge_lyrics  # 合并所有歌词


def spider():
    agent = random.choice(agents)  # 随机从爬虫头库中随机选取一个头，防止网站拦截
    header = {"User-Agent": agent}
    num = len(music_id_set)  # 获取总歌曲数量
    song_path = "../dataset/lyric/"  # 歌词保存路径

    if os.path.exists(song_path):  # 判断是否存在歌词文件夹，若存在则跳过
        print('已存在保存歌词文件夹，开始下载\n')
        time.sleep(1)  # 暂停一秒，然后开始下载歌词
        pass
    else:  # 若不存在，则创建
        os.mkdir("lyric")
        print('成功创建保存歌词文件夹')

    for i in range(num):
        lrc_url = 'http://music.163.com/api/song/lyric?' + 'id=' + str(music_id_set[i]) + '&lv=1&kv=1&tv=-1'
        lyric = requests.get(url=lrc_url, headers=header, timeout=7)
        json_obj = lyric.text
        j = json.loads(json_obj)

        try:  # 防止部分歌曲没有歌词，这里引入一个异常
            lrc = j['lrc']['lyric']
            part = re.compile(r'\[.*\]')
            lrc = re.sub(part, "", lrc)  # 正则替换
            lrc = lrc.strip()
            print("正在保存--", music_name_set[i])
            f = open("../dataset/lyric" + "/" + "%s.txt" % music_name_set[i], "w", encoding='utf-8')
            try:  # 尝试写入文件
                f.write(lrc)
                f.close()
            except AttributeError as e1:
                print('insert error', str(e1))
        except KeyError as e2:
            print('insert error', str(e2))
    merge_lyrics()  # 调用 merge_all_lyric 下 merge_lyrics 函数，合并所有歌词


if __name__ == '__main__':
    spider()
