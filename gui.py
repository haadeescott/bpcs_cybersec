from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import numpy as np
import imageio
import os
import time
import math
import os
import BPCS
from pathlib import Path



class Application(Frame):
    def __init__(self, master=None):
        self.file_name = ''
        Frame.__init__(self, master)
        self.pack()
        self.create_widgets()

    def open_file(self):
        # inside window open file window, only PNG files are selected
        self.file_name = askopenfilename(filetypes=[('PNG File', '*.PNG')])
        
        # to store file name as vessel name in BPCS.py
        BPCS.vessel_name = self.file_name

        # deletes path and only extracts the filename
        filename_only = Path(self.file_name).stem

        # to change label text dynamically according to file name
        self.name_label['text'] = 'Name: ' + filename_only
        
        #  left image/photo is the picture that is uploaded 
        global left_img
        left_img = None
        global left_photo
        left_photo = None

        left_img = Image.open(self.file_name)

        # once file is uploaded, clear image in target canvas
        # if left_img != '':
        #     self.target_img_canvas.delete('all')
        #     self.target_img_canvas.create_text(150,100,fill="darkblue",font="Times 14 italic bold",
        #                 text="Target Img will be displayed here.")
        
        # to retrieve and display dimensions(in px) and size(in KB) of uploaded image
        width, height = left_img.size
        imageKBsize = os.path.getsize(self.file_name)/1000
        self.dimensions_label['text'] = 'Dimensions: ' + str(width) + 'px' + ' x ' + str(height) + 'px'
        self.size_label['text']= 'Size: ' + str(imageKBsize) + 'KB'

       

        # resize image
        scale_width = img_display_width / width
        scale_height = img_display_height / height
        
        # get smallest compression capability of image
        scale = min(scale_width, scale_height)
        new_width = math.ceil(scale * width)
        new_height = math.ceil(scale * height)

        # Image.NEAREST http://pillow.readthedocs.io/en/4.1.x/releasenotes/2.7.0.html
        # image resize takes a resampling argument which tells which filter to use for resampling
        # image.nearest takes in nearest pixel from input image
        left_img = left_img.resize((new_width, new_height), Image.NEAREST)
        left_photo = ImageTk.PhotoImage(left_img)

        self.left_img_canvas.create_image(img_display_width / 2, img_display_height / 2, anchor=CENTER,
                                          image=left_photo)


    def open_targetFile(self):
        # inside window open file window, only PNG files are selected
        self.file_nameTarget = askopenfilename(filetypes=[('PNG File', '*.PNG')])

        # to store file name as Target name in BPCS.py
        BPCS.target_name = self.file_nameTarget

        # deletes path and only extracts the filename
        filename_only = Path(self.file_nameTarget).stem

        # to change label text dynamically according to file name
        self.name_labelTarget['text'] = 'Name: ' + filename_only

        # image and photo must be global or image will not appear
       

        # right image/photo is the picture that is produced by program
        global right_img
        right_img = None
        global right_photo
        right_photo = None


        right_img = Image.open(self.file_nameTarget)
        
        # to retrieve and display dimensions(in px) and size(in KB) of uploaded image
        width, height = right_img.size
        imageKBsize = os.path.getsize(self.file_nameTarget)/1000
        self.dimensions_labelTarget['text'] = 'Dimensions: ' + str(width) + 'px' + ' x ' + str(height) + 'px'
        self.size_labelTarget['text']= 'Size: ' + str(imageKBsize) + 'KB'

       

        # resize image
        scale_width = img_display_width / width
        scale_height = img_display_height / height
        
        # get smallest compression capability of image
        scale = min(scale_width, scale_height)
        new_width = math.ceil(scale * width)
        new_height = math.ceil(scale * height)

        # Image.NEAREST http://pillow.readthedocs.io/en/4.1.x/releasenotes/2.7.0.html
        # image resize takes a resampling argument which tells which filter to use for resampling
        # image.nearest takes in nearest pixel from input image
        right_img = right_img.resize((new_width, new_height), Image.NEAREST)
        right_photo = ImageTk.PhotoImage(right_img)

        self.target_img_canvas.create_image(img_display_width / 2, img_display_height / 2, anchor=CENTER,
                                          image=right_photo)

    # call operation 1 == embedding target image into vessel image from BPCS.py
    def embedImage(self):
        BPCS.operation = 1

    def create_widgets(self):
            # functions are defined on top, now to pack the widgets onto GUI
            # LEFT FRAME of GUI 
            # ______________________________________________________________
            left_frame = Frame(self)
            left_frame.pack(side=LEFT)

            show_frame = Frame(left_frame)
            show_frame.pack(side=TOP)

            open_frame = Frame(show_frame)
            open_frame.pack(side=TOP)

            open_label = Label(open_frame, text='Open Vessel Image (PNG File):')
            open_label.pack(side=LEFT)

            # open file button
            open_button = Button(open_frame, text='<select>', command=self.open_file, bg="white", fg="green")
            open_button.pack(side=LEFT)

            # recover target image button
            recoverImg_button = Button(open_frame, text='Recover Target Image', bg="black", fg="white")
            recoverImg_button.pack(side=RIGHT)
            
            # details about vessel image
            LabelFrame = Frame(show_frame)
            LabelFrame.pack(side=LEFT)

            self.name_label = Label(LabelFrame, text='Name: ')
            self.name_label.pack(side=TOP)

            self.dimensions_label = Label(LabelFrame, text='Dimensions: ')
            self.dimensions_label.pack(side=TOP)

            self.size_label = Label(LabelFrame, text='Size: ')
            self.size_label.pack(side=TOP)
            # ============================

            # ============================
            
            canvas_frame = Frame(left_frame)
            canvas_frame.pack(side=BOTTOM)

            self.left_img_canvas = Canvas(canvas_frame, bg='grey', width=img_display_width, height=img_display_height)
            self.left_img_canvas.create_text(150,100,fill="darkblue",font="Times 14 italic bold",
                        text="Vessel Img will be displayed here.")
            self.left_img_canvas.pack(side=LEFT)

            self.stego_img_canvas = Canvas(canvas_frame, bg='grey', width=img_display_width, height=img_display_height)
            self.stego_img_canvas.create_text(150,100,fill="darkblue",font="Times 14 italic bold",
                        text="Stego Img will be displayed here.")
            self.stego_img_canvas.pack(side=RIGHT)
            
            # RIGHT part of GUI
            # ______________________________________________________________
            right_frame = Frame(self)
            right_frame.pack(side=RIGHT)

            showright_frame = Frame(right_frame)
            showright_frame.pack(side=TOP)

            # insert open file button, label, etc. to allow user to choose which image to embed
            open_labelTarget = Label(showright_frame, text='Open Target Image (PNG File):')
            open_labelTarget.pack(side=LEFT)

            open_buttonTarget = Button(showright_frame, text='<select>', command=self.open_targetFile, bg="white", fg="green")
            open_buttonTarget.pack(side=LEFT)

            embedImg_button = Button(showright_frame, text='Embed Target Image', bg="white", fg="black")
            embedImg_button.pack(side=LEFT)

            self.name_labelTarget = Label(right_frame, text='Name: ')
            self.name_labelTarget.pack(side=TOP)

            self.dimensions_labelTarget = Label(right_frame, text='Dimensions: ')
            self.dimensions_labelTarget.pack(side=TOP)

            self.size_labelTarget = Label(right_frame, text='Size: ')
            self.size_labelTarget.pack(side=TOP)
            #=====================================================================



            # TARGET image canvas
            self.target_img_canvas = Canvas(right_frame, bg='grey', width=img_display_width, height=img_display_height)
            self.target_img_canvas.create_text(150,100,fill="darkblue",font="Times 14 italic bold",
                        text="Target Img will be displayed here.")
            self.target_img_canvas.pack(side=RIGHT)



left_img = None
left_photo = None
right_img = None
right_photo = None
stego_img = None
stego_photo = None

# default image GUI image size for the uploaded and produced image
img_display_width = 300
img_display_height = 200
app = Application()
app.master.title('My Steganography')
app.mainloop()