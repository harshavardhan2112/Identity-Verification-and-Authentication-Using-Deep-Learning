import tkinter
import PIL.Image, PIL.ImageTk
from tkinter import simpledialog
import time
from tkinter import messagebox
import os
from tkinter import *
import random
from datetime import date
from PIL import Image
import numpy as np
import cv2
import torch
import smtplib
import os
import face_recognition
from tkinter import messagebox
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

class App:
    global img_canvas
    global cascPath
    global faceCascade
    global tf1, tf2
    global capture_img
    def __init__(self, window, window_title, video_source=0):
        self.model = torch.hub.load('yolov5', 'custom', path='model/best.pt', force_reload=True,source='local')
        self.window = window
        self.window.title("Identity Verification and Authentication")
        self.window.geometry("1300x1200")
        self.video_source = video_source
        self.vid = MyVideoCapture(self.video_source)
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        self.font1 = ('times', 13, 'bold')
        self.capture_img = None
        self.encodings = np.load("model/encoding.npy")
        self.names = np.load("model/names.npy")

        font = ('times', 16, 'bold')
        title = Label(window, text='Identity Verification and Authentication')
        title.config(bg='darkviolet', fg='gold')  
        title.config(font=font)           
        title.config(height=3, width=120)       
        title.place(x=0,y=5)                  

        self.l1 = Label(window, text='Warning Msg: ')
        self.l1.config(font=self.font1)
        self.l1.place(x=50,y=500)

        self.tf1 = Entry(window,width=30)
        self.tf1.config(font=self.font1)
        self.tf1.place(x=250,y=500)

        self.l2 = Label(window, text='Roll Number')
        self.l2.config(font=self.font1)
        self.l2.place(x=50,y=550)

        self.tf2 = Entry(window,width=60)
        self.tf2.config(font=self.font1)
        self.tf2.place(x=250,y=550)

        self.l3 = Label(window, text='Email ID:')
        self.l3.config(font=self.font1)
        self.l3.place(x=50,y=600)

        self.tf3 = Entry(window,width=60)
        self.tf3.config(font=self.font1)
        self.tf3.place(x=250,y=600)

        
        self.btn_snapshot=tkinter.Button(window, text="Detect Card & Recognize Person", command=self.cardDetection)
        self.btn_snapshot.place(x=50,y=650)
        self.btn_snapshot.config(font=self.font1)
        
        self.btn_train=tkinter.Button(window, text="Send Message", command=self.sendMessage)
        self.btn_train.place(x=310,y=650)
        self.btn_train.config(font=self.font1)

        self.cascPath = "model/haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
               
        self.delay = 15
        self.update()
        self.window.config(bg='turquoise')
        self.window.mainloop()           

    def cardDetection(self):
        global capture_img
        option = 0
        ret, frame = self.vid.get_frame()
        img = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray,1.3,5)
        print("Found {0} faces!".format(len(faces)))
        for (x, y, w, h) in faces:
            img = cv2.resize(frame, (512, 512))
            results = self.model(img)
            results.xyxy[0]  # im predictions (tensor)
            out = results.pandas().xyxy[0]  # im predictions (pandas)
            if len(out) > 0:
                xmin = int(out['xmin'].ravel()[0])
                ymin = int(out['ymin'].ravel()[0])
                xmax = int(out['xmax'].ravel()[0])
                ymax = int(out['ymax'].ravel()[0])
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                self.tf1.delete(0,END)
                self.tf1.insert(0,"Card Detected")
                cv2.imshow("Capture Face",img)
                cv2.waitKey(0)
            else:
                small_frame = cv2.resize(frame, (800, 800))
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB color space
                face_locations = face_recognition.face_locations(rgb_small_frame)  # Locate faces in the frame
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  # Encode faces in the frame
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.encodings, face_encoding)  # Compare face encodings
                    face_distance = face_recognition.face_distance(self.encodings, face_encoding)  # Calculate face distance
                    best_match_index = np.argmin(face_distance)  # Get the index of the best match
                    if matches[best_match_index]:  # If the face is a match
                        name = self.names[best_match_index]  # Get the corresponding name
                        if name in self.names:
                            self.tf1.delete(0,END)
                            self.tf1.insert(0,"Card Not Detected")
                            self.tf2.delete(0,END)
                            self.tf2.insert(0,name)
                        else:
                            self.tf1.delete(0,END)
                            self.tf1.insert(0,"Card Not Detected")
                            self.tf2.delete(0,END)
                            self.tf2.insert(0,"Unable to Recognize")
    
    def sendMessage(self):
        email = self.tf3.get()
        rollno = self.tf2.get()
        print(email+" "+rollno)
        em = [];
        em.append(email)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as connection:
            email_address = 'harshakoramati111@gmail.com'
            email_password = 'alwaysharsha@123'
            connection.login(email_address, email_password)
            connection.sendmail(from_addr="harshakoramati111@gmail.com", to_addrs=em, msg="Subject : Warning Mail\n\nRoll No : "+rollno+" Not wearning ID Card")
        messagebox.showinfo("Email Sent", "Email Sent")            
        
    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            self.window.after(self.delay, self.update)
            
 
class MyVideoCapture:
    def __init__(self, video_source=0):
        
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.pid = 0
 
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
 
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
App(tkinter.Tk(), "Tkinter and OpenCV")
