# coding=utf-8
import tkinter as tk
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

from predict import  predict

# 创建窗口 设定大小并命名
window = tk.Tk()
window.title('垃圾分类')
window.geometry('1050x500')
global img_png  # 定义全局变量 图像的
var = tk.StringVar()  # 这时文字变量储存器

mainframe = ttk.Frame(window, padding="5 4 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)


def openImg():
    global img_png
    var.set('已打开')
    Img = Image.open('testImg/paper1.jpg')
    word = predict(Img)
    print(word)
    img_png = ImageTk.PhotoImage(Img)
    label_Img2 = tk.Label(image=img_png).grid(column=2, row=2, sticky=W)


num = 1


def change():  # 更新图片操作
    global num
    var.set('已预测')
    num = num + 1
    if num % 3 == 0:
        ttk.Label(mainframe,text= 'paper').grid(column=50, row=5, sticky=W)
        url1 = "results/result.png"
        pil_image = Image.open(url1)
        img = ImageTk.PhotoImage(pil_image)
        label_img.configure(image=img)
    window.update_idletasks()  # 更新图片，必须update


# row = 1
# epochs = StringVar()
# ttk.Label(mainframe, text="epochs:").grid(column=1, row=1, sticky=W)
# addr_entry = ttk.Entry(mainframe, width=7, textvariable=epochs)
# addr_entry.grid(column=2, row=1, sticky=(W, E))
#
# ttk.Button(mainframe, text="Train").grid(column=4, row=1, sticky=W)

# row = 2
ttk.Button(mainframe, text="打开", command=openImg).grid(column=1, row=2, sticky=W)
ttk.Button(mainframe, text="预测", command=change).grid(column=3, row=2, sticky=W)

# row = 3 创建文本窗口，显示当前操作状态
ttk.Label(mainframe, text="状态").grid(column=1, row=3, sticky=W)
ttk.Label(mainframe, textvariable=var).grid(column=3, row=3, sticky=W)

url = "testImg/logo.png"
pil_image = Image.open(url)
img = ImageTk.PhotoImage(pil_image)
label_img = ttk.Label(window, image=img, compound=CENTER)
label_img.grid(column=0, row=2, sticky=W)

# 运行整体窗口
window.mainloop()