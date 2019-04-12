# coding=utf-8
import os
from flask import Flask, request, render_template
from cnn_model import my_prediction_3_classifier

# 指定服务器IP
server_ip = '114.212.85.29'
# 上传图片保存的路径
UPLOAD_FOLDER = './save_images'  # 暂存路径
# 模型目录与模型名称 绝对路径
model_dir1 = '/home/nicehija/PycharmProjects/flask_project/my_web/cnn_model/my_model_inceptionV3/3classifier_0andOthers/0_1'
model_dir2 = '/home/nicehija/PycharmProjects/flask_project/my_web/cnn_model/my_model_inceptionV3/3classifier_0andOthers/0_2'
model_dir3 = '/home/nicehija/PycharmProjects/flask_project/my_web/cnn_model/my_model_inceptionV3/3classifier_0andOthers/0_3'
model_name1 = 'model.ckpt-6000'
model_name2 = 'model.ckpt-6000'
model_name3 = 'model.ckpt-6000'
# 允许上传图片格式
ALLOWED_EXTENSIONS = ('jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG')

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def hello_user():
    return render_template('start.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            photo_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(photo_path)
            # 函数返回的一个list,包含了4类全部预测按概率大小排序，其中每一类是一个tuple
            my_results = my_prediction_3_classifier.predict(photo_path, model_dir1, model_name1,
                                                            model_dir2, model_name2,
                                                            model_dir3, model_name3)
            os.remove(photo_path)
            return render_template('upload.html', photo_name=file.filename, results=int(my_results))
        else:
            return '<p> 你上传了不允许的文件类型 </p>'


if __name__ == '__main__':
    app.run(host=server_ip)
