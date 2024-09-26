import concurrent
from concurrent.futures import ThreadPoolExecutor
from selenium.webdriver.chrome.service import Service

import requests
from bs4 import BeautifulSoup
import time

from embed_api import get_embed
from send_llm import send2llm
from selenium import webdriver
from selenium.webdriver.common.by import By

def get_html(url):
    service = Service()
    driver = webdriver.Chrome()
    driver.get(url)

    time.sleep(2)
    try:
        driver.switch_to.frame('mainFrame')
        try:
            html = driver.find_element(By.CSS_SELECTOR, '#post-area').text
        except Exception as e:
            html = driver.find_element(By.TAG_NAME, 'body').text
    except:
        html = driver.find_element(By.TAG_NAME, 'body').text

    driver.close()
    return html





def crawl_and_summarize(keyword):
    blog_url = f"https://search.naver.com/search.naver?ssc=tab.blog.all&sm=tab_jum&query={keyword}"
    news_url = f"https://search.naver.com/search.naver?ssc=tab.news.all&where=news&sm=tab_jum&query={keyword}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(blog_url, headers=headers)
    response2 = requests.get(news_url, headers=headers)
    # 블로그와 뉴스 크롤링 대상 링크 수집
    if response.status_code != 200:
        return "Error fetching the page", 500
    blog_urls = []
    blog_soup = BeautifulSoup(response.text, 'html.parser')
    for item in blog_soup.select(
            'div#main_pack > section > div.api_subject_bx > ul > li > div > div.detail_box > div.title_area > a'):
        blog_urls.append(item.attrs['href'])
    news_soup = BeautifulSoup(response2.text, 'html.parser')
    news_url = []
    for item in news_soup.select('div.group_news > ul > li > div > div > div.news_contents > a.news_tit'):
        news_url.append(item['href'])
    # html 추출
    urls = blog_urls[:3] + news_url[:3]
    summarizes = [None] * len(urls)
    htmls = []
    for url in urls:
        html = get_html(url)[:16000]
        htmls.append(html)
    print("html 추출 완료, llm 전송 시작")
    def task(index, html):
        prompt = [{"role": "user", "content": "아래 웹페이지의 내용을 요약해 주세요. 작성자의 정보는 제외하고, 광고나 관련 없는 정보도 무시해 주세요."+
                                      "핵심 정보만 포함해 주세요. 영어로 생각하고 한글로 답변해주세요"+
        f"{keyword}과 관련없는 내용은 광고이다. 웹페이지 전체가 관련이 없다면 광고라고 판단하고 'advertisement' 라고 답변해라. "
                                              + html}, {"role": "system", "content":
            "You are an assistant that specializes in summarizing text. Focus on extracting key points and main ideas from the provided content. plese reply in korean"}]
        summarizes[index] = send2llm(prompt)
        print(summarizes[index])
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(task, i, obj): i for i, obj in enumerate(htmls)}
        for future in concurrent.futures.as_completed(futures):
            future.result()
    return summarizes





if __name__ == '__main__':
    summarizes = crawl_and_summarize("vllm")
    embeds = []
    for summary in summarizes:
        embeds += get_embed(summary)

