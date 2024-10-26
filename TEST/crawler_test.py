from crawler import naver_serch, search2naver
from embed_api import get_embed

test_target = {
    "naver_search" : False,
    "search2naver" : True
}


if __name__ == '__main__':
    if test_target["naver_search"]:
        summarizes = naver_serch("도로 정비 사업")
        print(summarizes)
    if test_target["search2naver"]:
        print(search2naver("도로 정비 사업"))