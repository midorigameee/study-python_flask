from flask import Flask, render_template, request, url_for, redirect
import os
import datetime


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


@app.route('/detail', methods=['GET'])
def detail():
    filename = request.args["filename"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    print("filename : " + filename)
    print("filepath : " + filepath)

    if os.name == "nt":
        print("For Windows:")
        filepath = converUrlForHtml(filepath)
        print("filepath : " + filepath)

    path_for_info = os.path.join("static", filepath)

    updated_time = getFileUpdatedTime(path_for_info)
    created_time = getFileCreatedTime(path_for_info)

    print("updated_time : {}".format(updated_time))
    print("created_time : {}".format(created_time))

    return render_template(
        "detail.html",
        filename=filename,
        filepath=filepath,
        updated_time=updated_time,
        created_time=created_time
        )


@app.route('/delete', methods=['GET'])
def delete():
    filename = request.args["filename"]

    filepath = os.path.join("static", app.config["UPLOAD_FOLDER"], filename)
    print("[DELETE]filepath : {}".format(filepath))

    os.remove(filepath)

    if os.path.exists(filepath) is None:
        print("Deleted [{}]".format(filepath))

    return render_template(
        "index.html",
        title="Top Page"
    )


def getFileUpdatedTime(filename):
    # 参考 https://www.mathpython.com/file-date
    t = os.path.getmtime(filename)  # この時点ではUNIX時間
    d = datetime.datetime.fromtimestamp(t)  # 一般の時間に変換

    return d


def getFileCreatedTime(filename):
    if os.name == "nt": # Win用
        t = os.path.getctime(filename)
        d = datetime.datetime.fromtimestamp(t)
    else:   # Mac用（Linuxは分からん、、、）
        t = os.stat(filename).st_birthtime
        d = datetime.datetime.fromtimestamp(t)

    return d


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