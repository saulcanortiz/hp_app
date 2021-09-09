from app import app
from flask import render_template, url_for, request, redirect
from flask_mail import Mail, Message
import os, cv2, glob
from NetManager import NetManager
import shutil





uploads_dir = os.path.join(app.instance_path,'video')
os.makedirs(uploads_dir, exist_ok=True)

def video2image(video_filename):
    video = cv2.VideoCapture(video_filename)
    sucess, image = video.read()
    count = 100
    while sucess:
        cv2.imwrite("video2img/frame%d.jpg" % count, image)
        sucess, image = video.read()
        count += 1
def video_mod(video_filename):
    img_array = []
    for filename in glob.glob('./results/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
 
    out = cv2.VideoWriter('video_mod.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    shutil.move("video_mod.mp4", "app/static/images")


@app.route("/")
def index():
    return render_template("public/index.html")

@app.route("/about")
def about():
    return render_template("public/about.html")

@app.route("/experience", methods=["GET","POST"])
def upload_file():
    if request.method == "POST":
        file = request.files['file']
        path = os.path.join(uploads_dir, file.filename)
        file.save(path)
        video2image(path)
        net = NetManager()
        net.load("bestModel_validationDEF.pt")
        net.test('./video2img','./results')
        video_mod('./results')
        return redirect(url_for("download_file"))
    return render_template("public/experience.html")

@app.route("/download_file", methods=["GET"])
def download_file():
    if request.method=="GET":
        return render_template("public/video.html")
