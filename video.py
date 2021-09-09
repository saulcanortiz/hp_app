import glob, cv2, shutil
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