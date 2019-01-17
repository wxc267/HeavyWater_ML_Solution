from flask import Flask, request, render_template, jsonify
from model import predict_result
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    words = request.args.get('words')
    print(words)
    result = predict_result.predict(words)
    return jsonify({'result':result})

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
