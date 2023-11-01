from flask import Flask, render_template, request, url_for, redirect
import os


app = Flask(__name__, static_folder='./static')

IMG_FOLDER = os.path.join("static", "images")
app.config["UPLOAD_FOLDER"] = IMG_FOLDER

@app.route('/')
def index():
    return render_template(
        "index.html",
        title="Top Page"
        )


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # URLでhttp://127.0.0.1:5000/uploadを指定したときはGETリクエストとなるのでこっち
    if request.method == 'GET':
        return render_template('upload.html')
    # formでsubmitボタンが押されるとPOSTリクエストとなるのでこっち
    elif request.method == 'POST':
        file = request.files['example']
        file.save(os.path.join('static', 'images', file.filename))
        return redirect(url_for('uploaded_file', filename=file.filename))


@app.route('/uploaded_file/<string:filename>')
def uploaded_file(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    return render_template('uploaded_file.html', filename=filename, filepath=filepath)


if __name__ == "__main__":
    app.run()
    # app.run(host='localhost')   # ローカル開発環境用の設定