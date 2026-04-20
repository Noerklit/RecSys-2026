from flask import Flask
import requests
app = Flask(__name__)

@app.route('/')
def hello():
    message = requests.get('http://my_backend:5000')
    return message.text

if __name__ == "__main__":
    app.run(host='0.0.0.0')
