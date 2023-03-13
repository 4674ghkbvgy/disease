from bs4 import BeautifulSoup
# 打开hand.html文件并解析为一个BeautifulSoup对象
with open("hand.html", "r", encoding="utf-8") as f:
soup = BeautifulSoup(f, "html.parser")

# 定位到所有的&lt;img&gt;标签，并获取它们的src属性
img_tags = soup.find_all("img")
src_list = [tag["src"] for tag in img_tags]

# 在每个src属性前面拼接上"https://dermnetnz.org"
full_src_list = ["https://dermnetnz.org" + src for src in src_list]

# 打印出结果
print(full_src_list)