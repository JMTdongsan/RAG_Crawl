from flask import Flask, request, jsonify

from crawler import crawl_and_summarize

app = Flask(__name__)



@app.route('/api/crawl', methods=['GET'])
def crawl_api():
    keyword = request.args.get('keyword')
    if not keyword:
        return jsonify({"error": "Keyword is required"}), 400

    summaries = crawl_and_summarize(keyword)
    return jsonify({"keyword": keyword, "summaries": summaries})
