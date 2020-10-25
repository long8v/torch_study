# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import requests
from collections import defaultdict
# import BeautifulSoup

def download(url="https://www.google.com/search",params={},retries=3):
    
    resp=None
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.108 Safari/537.36"
    header={"user-agent":user_agent}
    try:
        resp=requests.get(url,params=params,headers=header)
        resp.raise_for_status() #error/event 발생하면 except로 가게해라 
    except requests.exceptions.HTTPError as e:
        if e.response.status_code//100==5 and retries>0:
            print(retries)
            resp=download(url,params,retries-1)
        else: 
            print(e.response.status_code)
            print(e.response.reason)
            print(e.response.headers)
    return resp


threshold = 3
def urlExtractor(seed, limit):
    if limit > threshold:
        return []
    html = download(seed) 
    dom = BeautifulSoup(html.text, "lxml")
    links = [_["href"] for _ in dom.select("a") if _.has_attr("href")]
    unseen = list()
    for link in links:
        if len(link) > 1 and link[0] == "/":
            newLink = requests.compat.urljoin(seed, link)
            if newLink not in unseen:
                unseen.append((newLink, limit+1))
        elif link.startswith("http"):
            unseen.append((link, limit+1))
    return unseen
