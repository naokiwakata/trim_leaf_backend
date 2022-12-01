from flask import Flask
from flask_cors import CORS
from flask import request, make_response, jsonify

from predictor import LeafPredictor

import shutil
import os
import base64
import numpy as np
import cv2
import json

app = Flask(__name__)
CORS(app)

leafPredictor = LeafPredictor()


@app.route("/trimming", methods=['GET', 'POST'])
def trimming():
    # postされた画像を取得
    shutil.rmtree('static/imgs/')
    os.mkdir('static/imgs/')
    data = request.get_json()
    post_img = data['post_img']
    img_base64 = post_img.split(',')[1]
    img_binary = base64.b64decode(img_base64)
    img_array = np.asarray(bytearray(img_binary), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)

    # 葉を予測
    leafPredictor.predict(img=img)
    predicted_img = leafPredictor.predicted_img
    cv2.imwrite('result.jpg', predicted_img)

    # 保存した画像を開く
    with open('result.jpg', "rb") as f:
        predicted_img_base64 = base64.b64encode(f.read()).decode('utf-8')

    # 画像を切り抜く
    leafPredictor.createTrimedImg()
    cut_img_paths = leafPredictor.img_paths
    cut_img_base64_list = []

    for img_path in cut_img_paths:
        with open(img_path, "rb") as f:
            cut_img_base64 = base64.b64encode(f.read()).decode('utf-8')
            cut_img_base64_list.append(cut_img_base64)

    # 返す
    response = {
        'cut_imgs': cut_img_base64_list,
        'predict': predicted_img_base64, }
    
    dump = json.dumps(response)
    return make_response(dump)


if __name__ == "__main__":
    app.debug = True
    app.run(host='127.0.0.1', port=5000, threaded=True)
