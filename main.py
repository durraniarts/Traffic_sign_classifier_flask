from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from app import predict_image 


app = Flask(__name__)
app.config['IMAGE_UPLOADS'] = '/'
cors = CORS(app, origins='*')

# CORS(app, origins='*')
@app.route("/", methods=['POST'])
def hello_world():
    req = request.json
    val = req
    # print(val['data'][0]['data_url'])
    data_url = val['data'][0]['data_url']
    if data_url.startswith('data:image/jpeg;base64,'):
        pass
        # print(base64.b64decode(val['data'][0]['data_url'].split(',')[1]))
    image_data = base64.b64decode(val['data'][0]['data_url'].split(',')[1])
    with open("uploaded_image.jpg", "wb") as f:
        # print(f)
        f.write(image_data)
    res = predict_image("uploaded_image.jpg")
        # print(res)
        
        
   
    return jsonify(res)