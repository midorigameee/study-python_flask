from flask import Flask, render_template, request, url_for, redirect
import os


app = Flask(__name__, static_folder="static")
app.config["UPLOAD_FOLDER"] = "images"

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
    print("filename : " + filename)
    print("filepath : " + filepath)

    if os.name == "nt":
        print("For Windows:")
        filepath = converUrlForHtml(filepath)
        print("filepath : " + filepath)

    return render_template('uploaded_file.html', filename=filename, filepath=filepath)


@app.route('/confirm', methods=['GET'])
def confirm():
    filename_list = os.listdir(os.path.join("static", app.config["UPLOAD_FOLDER"]))
    print("filename_list : ")

    filename_list = checkImgList(filename_list)
    print(filename_list)


    filepath_list = []

    for fname in filename_list:
        fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)

        if os.name == "nt":
            print("For Windows:")
            fpath = converUrlForHtml(fpath)
        
        filepath_list.append(fpath)

    print("filepath_list : ")
    print(filepath_list)

    return render_template(
        'confirm.html',
        list_len=len(filename_list),
        filename_list=filename_list,
        filepath_list=filepath_list
    )


def checkImgList(img_list):
    checked_list = []

    for i, img in enumerate(img_list):
        if os.path.splitext(img)[1] == '.jpg':  
            checked_list.append(img)

    return checked_list


def converUrlForHtml(url):
    return url.replace("\\", "/")


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='localhost')   # ローカル開発環境用の設定