from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template(
        "index.html",
        title="Top Page"
        )

if __name__ == "__main__":
    # app.run(host='0.0.0.0')
    app.run(host='localhost')   # ローカル開発環境用の設定