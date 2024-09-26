import time
from concurrent.futures import ThreadPoolExecutor
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv('VLLM_URL'),
    api_key=os.getenv('vLLM_API_KEY'),
)

def send2llm(messages=None, model=os.getenv('DEFAULT_MODEL')):
    if messages is None:
        messages = [{"role": "user", "content": "who are you? "}]
    try:
        completion = client.chat.completions.create(model=model, messages=messages)
    except Exception as e:
        return "error :" + str(messages)
    return completion.choices[0].message

if __name__ == "__main__": # for test

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=100) as executor:
        for i in range(400):
            time.sleep(0.1)
            executor.submit(send2llm,messages=[{"role":"system","content":"you are a smart assistant."},
        {"role": "user", "content": "what is political correctness? why this is important? why some people hate this? 이민자 문제에 대해서 어떻게 적용시킬지 생각해줘, 한글로 답 해줘"}])
    print(send2llm(messages=[{"role":"system","content":"you are a smart assistant."},
        {"role": "user", "content": "what is political correctness? why this is important? why some people hate this? 이민자 문제에 대해서 어떻게 적용시킬지 생각해줘, 한글로 답 해줘"}]))
    print("%s milliseconds" % round((time.time() - start_time) * 1000))