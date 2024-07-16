import requests
from bs4 import BeautifulSoup

headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36',
    }

# 替换对应链接
response = requests.get('https://v.docin.com/search.do?source_type=2&pid=&nkey=%E4%BA%A4%E8%B0%88%E5%A3%B0', headers=headers)
content = response.text
soup = BeautifulSoup(content, 'html.parser')
all_c = soup.findAll( class_='m-showlst-pic')
link_box = []
for c in all_c:
    a_tag = c.find('a')
    if a_tag:
        href = a_tag.get('href')
        link_f = 'https://v.docin.com' + href
        link_box.append(link_f)
# print( link_box )

i = 1
for link in link_box:
    response_t = requests.get( link, headers=headers )
    content_t = response_t.text
    soup_t = BeautifulSoup( content_t, 'html.parser' )
    c_all = soup_t.find( 'audio' )
    if c_all:
        data_url = 'https:' + c_all.get( 'src' )
        print( data_url )
        response_f = requests.get(data_url, headers=headers)
        if response_f.status_code == 200:

            # 指定保存路径
            with open(f'../test/talk_sound{i}.mp3', 'wb') as file:

                file.write( response_f.content )
            print(f'MP3 file downloaded{i}')
        else:
            print(f"Failed to retrieve MP3 file{i + 1}. Status code:", response.status_code)
    i += 1