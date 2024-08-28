from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)


def llm_request(topic, article):
    url = "http://117.16.136.152:5000/generate"
    prompt = "원하는 정보는 {topic}에 관한 것이다. 그 이외 정보를 걸러서 핵심적인 주제만 제공해달라. 내용 : {article}"
    data = {"text": prompt.format(topic=topic, article=article)}
    response = requests.post(url, json=data)
    response = response.json()['response']
    assistant_content = next(item['content'] for item in response[0]['generated_text'] if item['role'] == 'assistant')
    print(response)
    return response

def summarize_all(article):
    url = "http://117.16.136.152:5000/generate"
    prompt = "중복되는 내용을 없애고 정보를 정리해줘 {article}"
    data = {"text": prompt}


def crawl_and_summarize(keyword):
    blog_url = f"https://search.naver.com/search.naver?ssc=tab.blog.all&sm=tab_jum&query={keyword}"
    news_url = f"https://search.naver.com/search.naver?ssc=tab.news.all&where=news&sm=tab_jum&query={keyword}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(blog_url, headers=headers)
    response2 = requests.get(news_url, headers=headers)

    if response.status_code != 200:
        return "Error fetching the page", 500
    blog_urls = []
    blog_soup = BeautifulSoup(response.text, 'html.parser')
    for item in blog_soup.select('d#main_pack > section > div.api_subject_bx > ul > li > div > div.detail_box > div.title_area > a'):
        blog_urls += item.attrs['href']
    news_soup = BeautifulSoup(response2.text, 'html.parser')
    news_url = []
    for item in news_soup.select('div.group_news > ul > li > div > div > div.news_contents > a.news_tit'):
        news_url.append(item['href'])

    summarizes = []
    for url in blog_urls+news_url:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        summarize = llm_request(keyword,soup.text.replace("\n\n","").replace("\n"," "))
        summarizes.append(summarize)
    return summarize_all(summarizes)


@app.route('/api/crawl', methods=['GET'])
def crawl_api():
    keyword = request.args.get('keyword')
    if not keyword:
        return jsonify({"error": "Keyword is required"}), 400

    summaries = crawl_and_summarize(keyword)
    return jsonify({"keyword": keyword, "summaries": summaries})


if __name__ == '__main__':
    app.run(debug=True)