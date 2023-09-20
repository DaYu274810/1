import torch
import ttkbootstrap as ttk
from torchvision import transforms
from ttkbootstrap.constants import *
from tkinter.messagebox import *
import tkinter as tk
import tkinter.filedialog
from PIL import Image,ImageTk
import tkinter.filedialog
from PIL import Image
import cv2 as cv
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage import  exposure
from tkinter import Scrollbar, Canvas

from vgg13.Vgg_13 import Vgg_13
from vgg13.mydataset import BirdDataset

#选择并显示图片
labelShowImages = []  # 存储图片标签的列表
# 用于存储图片对象的列表
image_objects = []


from tkinter import Scrollbar, Canvas

# 用于存储图片对象的列表
image_objects = []

def choosepic():
    path_ = tkinter.filedialog.askopenfilename(initialdir="E:/Python.Project/BirdIdenty/images")
    path.set(path_)
    print(path)
    img_open = Image.open(entry.get())
    img = ImageTk.PhotoImage(img_open)

    #使用Label显示图片
    lableShowImage = tk.Label(app,width=200,height=300)
    lableShowImage.pack()
    lableShowImage.place(x=100,y=300)
    tlabel1 = ttk.Label(app, text='原图',bootstyle="inverse",width=12,anchor=CENTER,font=("微软雅黑", 15))
    tlabel1.pack(side=ttk.LEFT, padx=5, pady=10)
    tlabel1.place(x=135,y=620)
    lableShowImage.config(image=img)
    lableShowImage.image = img

#均值滤波图片
def AverageValue():
    img1=cv.imread(entry.get())
    img1=cv.blur(img1,(3,3))
    cv.imwrite("./picture/average.jpg",img1)
    i1 = Image.open("./picture/average.jpg")
    img1 = ImageTk.PhotoImage(i1)
    # 使用Label1显示均值滤波图片
    lableShowImage1 = tk.Label(app, width=200, height=300)
    lableShowImage1.pack()
    lableShowImage1.place(x=400, y=300)
    tlabel2 = ttk.Label(app, text='均值滤波处理后',bootstyle="inverse",font=("微软雅黑", 15))
    tlabel2.pack(side=ttk.LEFT, padx=5, pady=10)
    tlabel2.place(x=430,y=620)
    lableShowImage1.config(image=img1)
    lableShowImage1.image = img1


#均值中波图片
def MidValue():
    img2 = cv.imread(entry.get())
    img2 = cv.medianBlur(img2,3)
    cv.imwrite("./picture/mid.jpg", img2)
    i2 = Image.open("./picture/mid.jpg")
    img2 = ImageTk.PhotoImage(i2)
    # 使用Label2显示中值滤波图片
    tlabel3 = ttk.Label(app, text='中滤波处理后',bootstyle="inverse",width=12,anchor=CENTER,font=("微软雅黑", 15))
    tlabel3.pack(side=ttk.LEFT, padx=5, pady=10)
    tlabel3.place(x=730,y=620)
    lableShowImage2 = tk.Label(app, width=200, height=300)
    lableShowImage2.pack()
    lableShowImage2.place(x=700, y=300)
    lableShowImage2.config(image=img2)
    lableShowImage2.image = img2

#Lbp特征提取
def showlbp():
    radius = 1  # LBP算法中范围半径的取值
    n_points = 8 * radius  # 领域像素点数

    # 读取图像
    lbpimage = cv.imread(entry.get())
    gray = cv.cvtColor(lbpimage, cv.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius)
    cv.imwrite("./feature/lbp.jpg", lbp)
    openlbp = Image.open("./feature/lbp.jpg")
    newlbp = ImageTk.PhotoImage(openlbp)
    # 使用lbpLabel显示LBP特征提取图片
    lbpLabel = tk.Label(app, width=200, height=300)
    lbpLabel.pack()
    lbpLabel.place(x=400, y=300)
    tlabel10 = ttk.Label(app, text='LBP特征提取',bootstyle="inverse",font=("微软雅黑", 15),width=12,anchor=CENTER)
    tlabel10.pack(side=ttk.LEFT, padx=5, pady=10)
    tlabel10.place(x=430,y=620)
    lbpLabel.config(image=newlbp)
    lbpLabel.image = newlbp


#HOG特征提取
def showhog():
    radius = 1  # LBP算法中范围半径的取值
    n_points = 8 * radius  # 领域像素点数

    # 读取图像
    hogimage = cv.imread(entry.get())
    gray = cv.cvtColor(hogimage, cv.COLOR_BGR2GRAY)
    fd,hog1=hog(gray,orientations=8,pixels_per_cell=(16,16), cells_per_block=(1,1),visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog1, in_range=(0, 0.02))
    final_img = gray * hog_image_rescaled
    cv.imwrite("./feature/hog.jpg",final_img)
    openhog = Image.open("./feature/hog.jpg")
    newhog = ImageTk.PhotoImage(openhog)
    # 使用hogLabel显示HOG特征提取图片
    hogLabel = tk.Label(app, width=200, height=300)
    hogLabel.pack()
    hogLabel.place(x=700, y=300)
    tlabel11 = ttk.Label(app, text='HOG特征提取',bootstyle="inverse",font=("微软雅黑", 15),width=12,anchor=CENTER)
    tlabel11.pack(side=ttk.LEFT, padx=5, pady=10)
    tlabel11.place(x=730,y=620)
    hogLabel.config(image=newhog)
    hogLabel.image = newhog

def identify():
    result= tk.Tk()
    result.title('结果')
    result.geometry("250x150+850+300")
    var=tk.StringVar(result,value='识别结果：\n\n1：\n\n2:\n\n')
    l=ttk.Label(result,textvariable=var,font=('Arial',15),width=400)
    l.pack()
    result.mainloop()
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }
    data_dir = './data1/CUB_200_2011/images/'
    model = Vgg_13()
    model.load_state_dict(torch.load('./vgg13/models/model.pt', map_location=lambda storage, loc: storage))
    model.train(False)
    model.eval()
    test_data = BirdDataset(data_dir=data_dir, filelist="./vgg13/test.txt", transform=data_transforms['test'])
    dataload = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    for data in dataload:
        inputs, labels = data
        outputs = model(inputs)  ##前向运行
        _, preds = torch.max(outputs.data, 1)
        data_sorted, idx_sorted = torch.sort(outputs, descending=True)  # 对输出结果进行排序，数值越大的表示越符合该标签表示的类
        ids = idx_sorted.numpy()
        print(ids[0][0])



if __name__ == '__main__':

    app = tk.Tk()
    app.title("鸟类识别")
    app.geometry("1000x700+150+50")
    app.attributes('-alpha', 1.0)

    background = Image.open("C:/Users/xqyy2/PycharmProjects/pythonProject5/1.webp")
    bk=ImageTk.PhotoImage(background)
    canvas = tk.Canvas(app, width=11000, height=50000)
    canvas.pack()
    canvas.create_image(0, 0, anchor='nw',image=bk)

    path = tk.StringVar()
    entry = tk.Entry(app, state='readonly', text=path,width =80)

    style=ttk.Style()
    style.configure("TButton",font=("微软雅黑", 15),padding=5)

    #
    flagLabel= ttk.Label(app, text='19200320余洋 ',bootstyle=DARK,anchor=CENTER,font=("微软雅黑", 20))
    flagLabel.pack(side=ttk.LEFT, padx=5, pady=10)
    flagLabel.place(x=420,y=80)
    infLabel= ttk.Label(app, text='鸟类识别',bootstyle=WARNING,anchor=CENTER,font=("微软雅黑", 44))
    infLabel.pack(side=ttk.LEFT, padx=5, pady=10)
    infLabel.place(x=400,y=0)
    infLabel = ttk.Label(app, text='降噪方式', bootstyle=PRIMARY, anchor=CENTER, font=("微软雅黑", 10))
    infLabel.pack(side=ttk.LEFT, padx=5, pady=10)
    infLabel.place(x=350, y=100)
    infLabel = ttk.Label(app, text='图像特征', bootstyle=INFO, anchor=CENTER, font=("微软雅黑", 10))
    infLabel.pack(side=ttk.LEFT, padx=5, pady=10)
    infLabel.place(x=625, y=100)

    # 选择图片的按钮
    button_00 = ttk.Button(app, text='选择图片', bootstyle=(SUCCESS, OUTLINE), width=9, command=choosepic)
    button_00.pack(side=LEFT, padx=5, pady=10)
    button_00.place(x=150, y=150)

    #按钮01均值滤波处理
    button_01 = ttk.Button(app, text='均值滤波', bootstyle=(PRIMARY,OUTLINE), width=9, command=AverageValue)
    button_01.pack(side=LEFT, padx=5, pady=10)
    button_01.place(x=350, y=150)

    # 按钮02中值滤波处理
    button_02 = ttk.Button(app, text='中值滤波', bootstyle=(PRIMARY,OUTLINE), width=9, command=MidValue)
    button_02.pack(side=LEFT, padx=5, pady=10)
    button_02.place(x=350, y=210)

    # 按钮03类型识别
    button_03 = ttk.Button(app, text='类型识别', width=9, command=identify, bootstyle=(DANGER,OUTLINE))
    button_03.pack(side=LEFT, padx=5, pady=10)
    button_03.place(x=750, y=150)


    # 按钮04lbp特征提取
    button_04 = ttk.Button(app, text='LBP', width=9, bootstyle=(INFO,OUTLINE), command=showlbp)
    button_04.pack(side=LEFT, padx=5, pady=10)
    button_04.place(x=550, y=150)

    # 按钮05hog特征提取
    button_05 = ttk.Button(app, text='HOG', width=9, bootstyle=(INFO,OUTLINE) ,command=showhog)
    button_05.pack(side=LEFT, padx=5, pady=10)
    button_05.place(x=550, y=210)


    app.mainloop()

