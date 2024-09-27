import time
from concurrent.futures import ThreadPoolExecutor
import os
from typing import List, Dict, Union, Any
from haystack import component
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from benchmark import bench_questions

client = OpenAI(
    base_url=os.getenv('VLLM_URL'),
    api_key=os.getenv('VLLM_API_KEY'),
)


def send2llm(messages: List[Dict[str, Union[str, Any]]]) -> str | ChatCompletionMessage:
    if messages is None:
        messages = [{"role": "user", "content": "who are you? "}]

    model = os.getenv('DEFAULT_MODEL')
    try:
        completion = client.chat.completions.create(model=model, messages=messages)
    except Exception as e:
        return "error :" + str(e)
    # print(completion.choices[0].message)
    return completion.choices[0].message


@component
class CustomGenerator:
    @component.output_types(replies=List[str])
    def run(self, prompt: str):
        # messages 형식으로 변환
        messages = [{"role": "user", "content": prompt}]
        # LLM 서버에 요청
        reply = send2llm(messages=messages)
        # LLM 서버의 응답에서 텍스트 추출
        if isinstance(reply, dict) and 'content' in reply:
            reply_content = reply['content']
        else:
            reply_content = str(reply)
        return {"replies": [reply_content]}


if __name__ == "__main__":  # for test
    futures = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=100) as executor:
        for i in range(400):
            time.sleep(0.1)
            messages = [{"role": "system", "content": "you are a smart assistant."},
                        {"role": "user", "content": bench_questions[i % len(bench_questions)]}]
            futures.append(executor.submit(send2llm, messages))

            # 모든 작업이 완료될 때까지 기다립니다
            for future in futures:
                print(future.result())
    print("%s milliseconds" % round((time.time() - start_time) * 1000))
