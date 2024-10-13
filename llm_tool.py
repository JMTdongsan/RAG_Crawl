import json

from bs4 import BeautifulSoup
import requests
import urllib.parse

from haystack import component

from config import client,model_name
from typing import Dict, Any, List
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from concurrent.futures import ThreadPoolExecutor, as_completed

from token_calc import truncate_to_max_tokens

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_word_definition",
            "description": "Get the definition of a word,only support korean,",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "The word to look up",
                    },
                },
                "required": ["word"],
            },
        }
    }
]
chatgpt_templete = {
        "type": "function",
        "function": {
            "name": "knowledge_page",
            "description": "Retrieve a detailed page that explains the concept, definition, or meaning of real estate investment terms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "The real estate term the user wants to understand, e.g., 'cap rate', 'cash flow', 'NOI (Net Operating Income)', '1031 exchange', etc."
                    }
                },
                "required": ["term"]
            }
        }
    }


@component
class ToolGenerator:
    @component.output_types(replies=List[str])
    def run(self, prompt: str):
        reply = vanila_inference(message=prompt)
        # LLM 서버의 응답에서 텍스트 추출
        if isinstance(reply, dict) and 'content' in reply:
            reply_content = reply['content']
        else:
            reply_content = str(reply)
        return {"replies": [reply_content]}


def fetch_and_summarize(url, word):
    """각 URL에서 데이터를 가져와 요약하는 함수"""
    full_url = "https://www.molit.go.kr" + url
    html = BeautifulSoup(requests.get(full_url).text, 'html.parser')
    body = html.select_one('#cont-body')
    body = str(body)
    completion = vanila_inference(f"아래의 글을 읽고 {word}에 관련한 내용이다. 아래의 글을 잘 이해가 되게 자세히 설명해라. "
                                  f"예시가 있다면 예시도 같이 설명해라. {body}")
    return completion


def get_word_definition(word: str):
    url =f"https://www.molit.go.kr/search/search2023.jsp?query={word}&collections=n_policy&listCount=10&startCount=0"
    encoded = urllib.parse.quote(urllib.parse.quote(word))
    url = f"https://www.molit.go.kr/wisenut/search.do?query={encoded}&collections=n_policy&listCount=10&startCount=0"
    print(url)
    html = BeautifulSoup(requests.get(url).text,'html.parser')
    data = json.loads(html.get_text())
    urls = [item['url'] for item in data['searchResult']['n_policy']]
    futures = []
    extract = ""
    results = []

    # ThreadPoolExecutor로 병렬 처리
    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        # 모든 Future 작업을 제출하고 관리
        futures = {executor.submit(fetch_and_summarize, url, word): url for url in urls}

        # as_completed로 Future의 결과 순차 처리
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error fetching data from {futures[future]}: {e}")

    # 모든 결과를 하나의 문자열로 합침
    extract = ''.join(results)
    message = f"아래의 글을 읽고 {word}에 대해서 최대한 자세히 설명해달라. 예시가 있다면 같이 설명해달라." + extract
    return vanila_inference(message)


def get_function_by_name(name:str):
    if name == "get_word_definition":
        return get_word_definition



def send2llm(messages: List[Dict[str, Union[str, Any]]]) -> str :
    if messages is None:
        messages = [{"role": "user", "content": "who are you? "}]
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            tools=tools ) # tools를 LLM에게 전달
    except Exception as e:
        return "error :" + str(e)
    # print(completion.choices[0].message)
    return completion.choices[0].message.content



def generate_function_call(messages, tools):
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
        extra_body={
            "repetition_penalty": 1.05,
        },
    )
    return response.choices[0].message


# 함수 실행 및 결과 추가
def execute_function_call(messages, tool_calls):
    for tool_call in tool_calls:
        call_id = tool_call["id"]
        if fn_call := tool_call.get("function"):
            fn_name = fn_call["name"]
            fn_args = json.loads(fn_call["arguments"])
            fn_res = json.dumps(get_function_by_name(fn_name)(**fn_args))
            messages.append({
                "role": "tool",
                "content": fn_res,
                "tool_call_id": call_id,
            })
    return messages


# 최종 응답 생성
def generate_final_response(messages, tools):
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
        extra_body={
            "repetition_penalty": 1.05,
        },
    )
    return response.choices[0].message.content


#Testing the function call
if __name__ == "__main__":
    MESSAGES = [
        {"role": "system",
         "content": "당신은 한국어 사전 기능을 갖춘 도우미입니다. 사용자가 단어나 구의 정의를 물어볼 때, 전체를 하나의 단위로 처리하세요. "
                    "특히 복합 용어(예: '관리 감독')는 개별 단어로 나누지 말고 전체를 하나의 검색어로 사용하세요."},
        {"role": "user", "content": "도로 정비 사업이 뭐지?"},
    ]
    messages = MESSAGES[:]

    # 함수 호출 생성
    function_call_message = generate_function_call(messages, tools)
    messages.append(function_call_message.model_dump())

    # 함수 실행 및 결과 추가
    if tool_calls := messages[-1].get("tool_calls", None):
        messages = execute_function_call(messages, tool_calls)

    # 최종 응답 생성
    final_response = vanila_inference(messages, tools)
    print("Assistant's response:", final_response)
