from flask import Flask, render_template, request, url_for, redirect
import os
import datetime
import cv2
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.autograd import Variable


app = Flask(__name__, static_folder="static")
app.config["UPLOAD_FOLDER"] = "images"
app.config["CASCADE_PATH"] = "cascade"
app.config["CASCADE_NAME"] = "haarcascade_frontalface_default.xml"
app.config["CLASSIFIER_PATH"] = "classifier"

@app.route('/')
def index():
    return render_template(
        "index.html"
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
    filepath = createImgPath(filename)

    print("[UPLOAD]filename : {}".format(filename))
    print("[UPLOAD]filepath : {}".format(filepath))

    return render_template(
        'uploaded_file.html',
        filename=filename,
        filepath=filepath
        )


@app.route('/view', methods=['GET'])
def view():
    # static/images以下のデータを全て取得する
    filename_list = checkImgExtension(os.listdir(os.path.join("static", app.config["UPLOAD_FOLDER"])))
    filepath_list = []
    for fname in filename_list:
        fpath = createImgPath(fname)
        filepath_list.append(fpath)

    print("[VIEW]filename_list : {}".format(filename_list))
    print("[VIEW]filepath_list : {}".format(filepath_list))

    return render_template(
        'view.html',
        list_len=len(filename_list),
        filename_list=filename_list,
        filepath_list=filepath_list
        )


@app.route('/detail', methods=['GET'])
def detail():
    filename = request.args["filename"]
    filepath = createImgPath(filename)

    print("[DETAIL]filename : " + filename)
    print("[DETAIL]filepath : " + filepath)

    path_for_info = os.path.join("static", filepath)
    updated_time = getFileUpdatedTime(path_for_info)
    created_time = getFileCreatedTime(path_for_info)

    print("[DETAIL]updated_time : {}".format(updated_time))
    print("[DETAIL]created_time : {}".format(created_time))

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
    filepath_for_py = os.path.join("static", app.config["UPLOAD_FOLDER"], filename)
    print("[DELETE]filename : {}".format(filename))
    print("[DELETE]filepath_for_py : {}".format(filepath_for_py))

    os.remove(filepath_for_py)

    if os.path.exists(filepath_for_py) is None:
        print("Delete completed [{}]".format(filepath_for_py))

    return render_template(
        "index.html"
        )


@app.route('/app_list', methods=['GET'])
def app_list():
    if "app_name" not in request.args:
        print("[EXECUTE APP]App name is not defined.")
        return render_template(
            "app_list.html"
            )
    
    print("[EXECUTE APP]request.args[app_name] : {}".format(request.args["app_name"]))

    if request.args["app_name"] == "actress_classify":
        return redirect(url_for('actress_classify'))
    else:
        print("[EXECUTE APP]App_name is not suitable.")
        return redirect(url_for('index'))


@app.route('/actress_classify', methods=['GET', 'POST'])
def actress_classify():
    if request.method == 'GET':
        filename_list = checkImgExtension(os.listdir(os.path.join("static", app.config["UPLOAD_FOLDER"])))
        filepath_list = []
        for fname in filename_list:
            fpath = createImgPath(fname)
            filepath_list.append(fpath)

        print("[ACTRESS CLASSIFY]filename_list : {}".format(filename_list))
        print("[ACTRESS CLASSIFY]filepath_list : {}".format(filepath_list))

        return render_template(
            "actress_classify.html",
            list_len=len(filename_list),
            filename_list=filename_list,
            filepath_list=filepath_list
            )
    
    elif request.method == 'POST':
        filename = request.form['selected_image']
        return redirect(url_for("actress_classify_result", filename=filename))


@app.route('/actress_classify_result/<string:filename>')
def actress_classify_result(filename):
    result = detect_face(filename)
    message = ""
    classifier_result = ""

    if result is None:
        message = "No face"
        result = "None"
    else:
        result = converUrlForHtml(result)
        classifier_result = useClassifier(face_path=os.path.join("static", "temp", "extracted_face.jpg"))

    return render_template(
        "actress_classify_result.html",
        result_path=result,
        message=message,
        classifier_result=classifier_result
        )


def detect_face(filename):
    file_path = os.path.join("static", app.config["UPLOAD_FOLDER"], filename)
    face_info = extract_maxsize_face(file_path)

    if face_info is None:
        print("画像が見つからないよ")
        return None

    if face_info["x"] == 0 and face_info["size"] == 0:
        print("顔が見つからないよ")
        return None

    # 画像の読み込み
    image = cv2.imread(file_path)

    # 顔の部分だけ抽出して保存
    extracted_face = image[face_info["y"]:face_info["y"]+face_info["size"], \
                            face_info["x"]:face_info["x"]+face_info["size"]]
    
    cv2.rectangle(image,
        (face_info["x"], face_info["y"]),   # 始点（左上）の座標
        (face_info["x"]+face_info["size"],face_info["y"]+face_info["size"]),    # 終点（右下）の座標
        (0, 255, 0), thickness=2
        )


    temp_dir = os.path.join("static", "temp")
    if os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
    else:
        os.mkdir(temp_dir)

    result_file = os.path.join(temp_dir, filename)
    extracted_face_file = os.path.join(temp_dir, "extracted_face.jpg")
    cv2.imwrite(result_file, image)
    cv2.imwrite(extracted_face_file, extracted_face)

    result_file_for_html = os.path.join("temp", filename)

    return result_file_for_html


def extract_maxsize_face(image_path):
    if os.path.exists(image_path) is None:
        print("image_path[{}] not exist.".format(image_path))
        return None

    image = cv2.imread(image_path)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join("static", app.config["CASCADE_PATH"], app.config["CASCADE_NAME"])
    cascade = cv2.CascadeClassifier(cascade_path)
    face_list = cascade.detectMultiScale(image_grey, minSize=(200, 200))

    face_info = {"x":0, "y":0, "size":0}
    for (x, y, w, h) in face_list:  # 顔は正方形で検出するので常にw=hになる
        # 一番大きい検出結果を検出対象とする
        if w > face_info["size"]:
            face_info["x"] = x
            face_info["y"] = y
            face_info["size"] = w

    return face_info


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


def checkImgExtension(img_list):
    checked_list = []

    for i, img in enumerate(img_list):
        if os.path.splitext(img)[1] == '.jpg':  
            checked_list.append(img)

    return checked_list


# ./image/filenameのパスを生成するメソッド
def createImgPath(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if os.name == "nt":
        filepath = converUrlForHtml(filepath)

    return filepath


# Windows環境でos.path.joinを使ってURL作成するとスラッシュがバックスラッシュになってしまうので修正するメソッド
def converUrlForHtml(url):
    return url.replace("\\", "/")


def useClassifier(face_path):
    # Hyper parameters
    FACE_SIZE = 32
    BATCH_SIZE = 10
    
    label = ["gacky", "kanna", "kasumi", "riho", "satomi", "tsubasa"]
    class_num = 6   # 6人の分類
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CNN_32(output_size=class_num).to(device)
    model_name = os.path.join("static", "classifier", "model_actress.ckpt")
    param = torch.load(model_name) # パラメータの読み込み
    model.load_state_dict(param)


    # ネットワークを推論モードに切り替える
    model.eval()
    face_image = cv2.imread(face_path)
    size = (FACE_SIZE, FACE_SIZE)
    face_image = cv2.resize(face_image, size)

    with torch.no_grad():

        # 学習済みモデルに抽出した顔画像を入力
        x = np.transpose(face_image, (2, 0, 1))  # (縦, 横, ch)を(ch, 縦, 横)
        x = np.array([x])
        x = torch.Tensor(x)
        x = Variable(x).to(device)

        # 認識結果
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        answer = predicted[0]

    return label[answer]

class CNN_32(nn.Module):
    def __init__(self, output_size):
        super(CNN_32, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='localhost')   # ローカル開発環境用の設定