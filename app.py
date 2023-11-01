from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, Python x Flask"

if __name__ == "__main__":
    # app.run(host='0.0.0.0')
    app.run(host='localhost')   # ローカル開発環境用の設定