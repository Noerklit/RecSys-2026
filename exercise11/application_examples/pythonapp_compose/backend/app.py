from flask import Flask
import requests
app = Flask(__name__)

@app.route("/")
def hello():
    return "You're the best!!"

@app.route("/myapi")
def cheerful_msg():
    return "You're the best!"

if __name__ == "__main__":
    app.run(host='0.0.0.0')
