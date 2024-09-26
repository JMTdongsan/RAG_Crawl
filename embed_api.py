import os

import requests
import json

# URL 설정 (API 엔드포인트)
url = os.getenv('EMBED_URL')  # 여기에 실제 URL을 입력하세요.
# sub_url = "http://117.16.136.198:8080/embed"

def get_embed(inputs):
    data = {"inputs": inputs}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        raise Exception(response.text)
    return response.json()[0]

if __name__ == "__main__": #for test
    embed = get_embed("소버린 AI(Sovereign AI)가 세계적 화두로 떠올랐다. 소버린 AI는 각 국가가 자체 데이터와 인프라를 활용해 그 국가나 지역의 제도, 문화, 역사, 가치관을 정확히 이해하는 AI를 개발하고 운영하는 걸 말한다. 소버린(sovereign)은 ‘자주적인’ ‘주권이 있는’ 이라는 뜻이다. 소버린 AI는 아무나 가질 수 없다. 고성능 그래픽 처리 장치를 보유한 데이터 센터와 이를 뒷받침하는 전력망, 데이터 수급, 실제 서비스에 적용하는 과정을 갖춰야 한다. 막대한 돈과 데이터, 기술, 인프라가 필요한 것이다. 실제 세계적으로 소버린AI를 갖고 있는 나라는 우리나라를 포함해 몇 곳 안된다.")
    print(embed)  # 1024개의 -1 ~ 1 까지의 fp32 형태의 리턴
