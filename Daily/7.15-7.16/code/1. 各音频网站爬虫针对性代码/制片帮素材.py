from bs4 import BeautifulSoup
import requests
import time

headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0',
    }

# 替换对应链接
response = requests.get('https://stock.zhipianbang.com/sound/search.html?keyword=%E8%BD%A6%E6%B5%81', headers = headers )

content = response.text
soup = BeautifulSoup( content, 'html.parser' )
all_h2 = soup.findAll( 'h2', class_='over_text1 font16' )
all_links = soup.findAll('a')
link_box = []
for h2 in all_h2:
    a_tag = h2.find('a')
    if a_tag:
        href = a_tag.get('href')
        link_f = 'https://stock.zhipianbang.com' + href
        link_box.append( link_f )
# print( link_box )

i = 1
for link in link_box:
    response_in = requests.get( link )
    content_in = response_in.text
    soup_in = BeautifulSoup( content_in, 'html.parser' )
    music_head = soup_in.find( class_ = 'music_head' )
    if music_head:
        data_url = music_head.get('data-url')
        print( data_url )
        response_f = requests.get( data_url, headers = headers )
        if response_f.status_code == 200:

            # 指定保存路径
            with open( f'../street_sound/street_sound{i}.mp3', 'wb') as file:

                file.write( response_f.content )
            print( f'MP3 file downloaded{i}' )
        else:
            print( f"Failed -> file{i + 1}file{i+1}. Status code:", response.status_code )
        time.sleep( 0.8 )
    i += 1

print( 'finished' )