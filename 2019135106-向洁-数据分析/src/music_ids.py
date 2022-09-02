"""获取所有歌词id"""

import requests  # 网页请求
import random
from bs4 import BeautifulSoup  # 网页解析

from user_agents import agents
from singer_id import all_singer_id_set

agent = random.choice(agents)  # 随机从爬虫头库中随机选取一个头，防止网站拦截
header = {"User-Agent": agent}
singer_id_set_num = len(all_singer_id_set)  # 计算所有歌手id个个数
music_id_set = []  # 放置歌曲id
music_name_set = []  # 放置歌曲名

for singer_id_set in range(singer_id_set_num):  # 获取歌词id
    # 防止报错（IndexError: list index out of range），引入try，except
    # https://blog.csdn.net/weixin_37746009/article/details/94367162
    try:
        singer_url = 'http://music.163.com/artist?id=' + str(all_singer_id_set[singer_id_set])  # 歌手所在网址
        web_data = requests.get(url=singer_url, headers=header, timeout=6)  # 自动请求页面，自动爬取HTML页面
        soup = BeautifulSoup(web_data.text, 'lxml')  # 通过 beautifulsoup库解析网页
        r = soup.find('ul', {'class': 'f-hide'}).find_all('a')  # 查找指定标签a，即歌曲名
        r = (list(r))  # 转换为列表
        for each in r:
            song_name = each.text_season
            music_name_set.append(song_name)
            song_id = each.attrs["href"]  # 获取属性为href内容，即歌曲id
            music_id_set.append(song_id[9:])  # href="/song?id=28310921"  选取href内容第九位数之后部分，
    except IndexError:
        print('IndexError')
        pass
print(len(music_id_set))
