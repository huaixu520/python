# 导包中的Flask相当于平台
from flask import Flask, render_template, request
# 导入base64
import base64
# 导入json模块，为前端传输base64编码数据
import json
# 导入detect.py中的YOLOV5类
from detect import YOLOV5
# 导入Pillow模块的Image类，帮助我们把ndarray转成图片
from PIL import Image
# 导入人脸识别模块
from face_recognition import load_image_file, compare_faces, face_encodings
import pymysql
import time

app = Flask(__name__, template_folder="my_html", static_folder="my_js")


# app表示的就是应用，route表示路由
# 这里的"/"表示根目录
@app.route("/")
def hello():
    # 返回值,"Hello world"作为文本在网页中显示
    return "Hello World"


@app.route("/index")
def index():
    return render_template("index1.html")


@app.route("/cut_face", methods=["GET", "POST"])
def cut_face():
    data = request.form.get("face")

    # 把数据保存成图片，前端返回的64编码数据，后端base64解码
    with open("face.jpg", "wb") as f:
        f.write(base64.b64decode(data))
    # 这里face-recognition也是机器学习完成的，接收数据是ndarray
    yolov5 = YOLOV5()
    result = yolov5.infer("face.jpg")
    x1 = result[0][0]
    y1 = result[0][1]
    x2 = result[0][2]
    y2 = result[0][3]
    names = result[1]
    mydata = load_image_file("result.jpg")
    cut_face = mydata[y1:y2, x1:x2, :]
    myimg = Image.fromarray(cut_face)
    myimg.save("result.jpg")
    # 把图片数据回传给前端，前端接收的是base64数据，现在cut_face是ndarray类型
    # 通过分析可以知道 result.jpg编译成base64编码，再往前端发送.
    with open("result.jpg", "rb") as f:
        img_result = base64.b64encode(f.read())
    # 把img_result的编码输出由unicode转成utf8,去掉输出的b
    img_result = img_result.decode("utf8")
    # 把time.localtime()初始化成"2022年6月29日 15:01:09"
    time_str = time.localtime()
    # time.strftime时间格式化不能接中文,输出格式变成"2022年6月29日 15:20:19"
    time_now = time.strftime("%Y{}%m{}%d{} %H:%M:%S", time_str).format("年", "月", "日")
    # 返回前端前把数据存到数据库中,导入pymysql
    conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456",
                           database="myface_code")
    # 获取数据库的游标
    cursor = conn.cursor()
    # 安全代码，把前面的记录先清除掉，只保留你最后一次的签到记录，
    # delete from faces 删除全部记录
    cursor.execute("delete from faces")
    # 再执行一遍提交
    conn.commit()
    # 由游标来执行sql语句,存储人脸数据的时候，存的应该是400*400*3
    cursor.execute("insert into faces(id,face,qiandao,qiantui) values(%s,%s,%s,%s)",
                   (1, data, time_now, None))
    # 提交结果是由连接来完成的
    conn.commit()
    # 最后关闭数据库
    conn.close()
    # 直接发送到前端，由前端自行处理base64的前22个字节。
    # 不管是前端发后端，还是后端发前端，都使用json数据
    # json转字符串用json.dumps
    names_result = "没有检测目标"
    if names == "mask":
        names_result = "已佩戴口罩"
    if names == "face":
        names_result = "没佩戴口罩"
    return json.dumps({"face": img_result, "time": time_now, "names": names_result}, ensure_ascii=False) # 保持编码不变


@app.route("/qiantui", methods=["GET", "POST"])
def qiantui():
    data = request.form.get("face")
    # 把数据保存成图片，前端返回的64编码数据，后端base64解码
    with open("face.jpg", "wb") as f:
        f.write(base64.b64decode(data))
    # 这里face-recognition也是机器学习完成的，接收数据是ndarray
    mydata = load_image_file("face.jpg")
    # 因为数据库存储的是base64编码，不需要转成ndarray，也不需要存图片,直接连接数据库
    # 连接数据库
    conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456",
                           database="myface_code")
    # 获取游标，把上一步连接的结果conn引入过来
    cursor = conn.cursor()
    # 执行数据库的逻辑
    # 1、把数据里面所有人脸数据全部查出
    cursor.execute("select face from faces")
    # 2、数据执行查询时,结果在cursor.fetchall()方法获取结果
    results = cursor.fetchall()
    for result in results:
        # 因为result结果是元素，元素只能取0元素，又因为有一个b,去掉b使用decode("utf8")
        # print(result[0].decode("utf8"))
        # 把查出的base64编码赋值给一个变量
        img_result = result[0].decode("utf8")
        # base64编码的img_result变成ndarray,必须先存储图片，再load_image
        with open("c.jpg", "wb") as f:
            # 因为存储图像时曾经encode过，所以调用base64.b64decode
            f.write(base64.b64decode(img_result))
        # 再次load_image_file就是ndarray
        myface = load_image_file("c.jpg")
        # compare_faces，第一个参数是[array([])],第二个参数只能array
        compare_result = compare_faces(face_encodings(mydata), face_encodings(myface)[0], 0.8)
        # 注意得到的结果是[True],只能取0元素
        if compare_result[0] == True:
            # 导入时间模块，获取当前时间
            # 把time.localtime()初始化成"2022年6月29日 15:01:09"
            time_str = time.localtime()
            # time.strftime时间格式化不能接中文,输出格式变成"2022年6月29日 15:20:19"
            time_qiantui = time.strftime("%Y{}%m{}%d{} %H:%M:%S", time_str).format("年", "月", "日")
            # 更新数据的时候，人脸数据是全部图像，同时img_result是base64编码
            cursor.execute("update faces set qiantui=%s where face=%s", (time_qiantui, img_result))
            # 数据执行增删改需要commit提交
            conn.commit()
    # #最后关闭数据库
    conn.close()
    # 这里返回json总报错，先直接把time_qiantui的时间返回
    return time_qiantui


# __name__=="__main__"，这句话意思表示__name__只有等于main才执行
# 主程序，但是清楚python中__name__会有很多内容
if __name__ == "__main__":
    app.run(port=8080)

