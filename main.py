from tkinter import *
from tkinter.ttk import *
import time
import tkinter as tk  # import tkinter
from tkinter import ttk
from csv import DictWriter
from tkinter import messagebox
from subprocess import *
import execnet
import os
import threading
from concurrent import futures
import time



class App:
    def __init__(self, parent):
        # create labels
        # name label
        self.name_label = ttk.Label(win, text="Name : ")
        self.name_label.grid(row=0, column=0, sticky=tk.W)


        # age label
        self.age_label = ttk.Label(win, text="Age : ")
        self.age_label.grid(row=1, column=0, sticky=tk.W)


        # gender label
        self.gender_label = ttk.Label(win, text="Gender : ")
        self.gender_label.grid(row=2, column=0, sticky=tk.W)

        # Create entry box
        # name entry box
        self.name_var = tk.StringVar()
        self.name_entrybox = ttk.Entry(win, width=16, textvariable=self.name_var)
        self.name_entrybox.grid(row=0, column=1)
        self.name_entrybox.focus()


        # age entry box
        self.age_var = tk.StringVar()
        self.age_entrybox = ttk.Entry(win, width=16, textvariable=self.age_var)
        self.age_entrybox.grid(row=1, column=1)

        # gender entry box
        # create combobox
        self.gender_var = tk.StringVar()
        self.gender_combobox = ttk.Combobox(win, width=13, textvariable=self.gender_var, state="readonly")
        self.gender_combobox['values'] = ('Male', 'Female', 'Other')
        self.gender_combobox.current(0)
        self.gender_combobox.grid(row=2, column=1)



        # create check button
        self.checkbtn_var = tk.IntVar()
        self.checkbtn = ttk.Checkbutton(win, text='Sensor is calibrated', variable=self.checkbtn_var)
        self.checkbtn.grid(row=6, columnspan=3)
        self.connect_text = tk.Text(win, relief="sunken")
        self.connect_text.grid(row=16, column=0)



        self.progress = Progressbar(win, orient=HORIZONTAL, length=100, mode='determinate')
        self.progress.grid(row=27, column=0)


        self.theButton= Button(win, text='Start', command =lambda: self.start_leap_in_bg()).grid(row=15, column=0)
        self.button2 = Button(win, text='Connect', command = lambda:[self.collect(),self.bar(3),self.image_change]).grid(row=15, column=50)
        # Get the current working directory
        cwd = os.getcwd()
        path = cwd + "\\pinch.png"
        image = tk.PhotoImage(file="pinch.png") ## Replace with gif
        self.label = tk.Label(image=image)
        self.label.grid(row=100, column=100)

        # train button
        self.train_button = ttk.Button(win, text="Train", command=self.action)
        self.train_button.grid(row=100, column=0)

        # test button
        self.test_button = ttk.Button(win, text="Test", command=self.action)
        self.test_button.grid(row=100, column=50)


        # classifier label
        self.classifier_label = ttk.Label(win, text="Classifier")
        self.classifier_label.grid(row=100, column=5, sticky=tk.W)

# Create button code action function
    def action():
        username = name_var.get()
        userage = age_var.get()
        usergender = gender_var.get()
        usertype = user_type.get()
        # change value 0,1 to Yes or No
        if checkbtn_var.get() == 0:
            subscribe = 'No'
        else:
            subscribe = 'Yes'

        # write to csv file code here
        with open('file.csv', 'a', newline='') as f:
            dict_writer = DictWriter(f, fieldnames=['User Name', 'User Age', 'User Email', 'User Mobile', 'User Gender',
                                                    'User Type', 'Subscribe'])
            if os.stat('file.csv').st_size == 0:  # if file is not empty than header write else not
                dict_writer.writeheader()

            dict_writer.writerow({
                'User Name': username,
                'User Age': userage,
                'User Email': useremail,
                'User Mobile': usermobile,
                'User Gender': usergender,
                'User Type': usertype,
                'Subscribe': subscribe
            })

    def bar(self,duration):
        self.progress['value'] = 20
        win.update_idletasks()
        time.sleep(duration/4)
        self.progress['value'] = 50
        win.update_idletasks()
        time.sleep(duration/4)
        self.progress['value'] = 80
        win.update_idletasks()
        time.sleep(duration/4)
        self.progress['value'] = 100

    def start_leap_in_bg(self):
        loop_thread = threading.Thread(target=connectDevice.leapConnect(self)).start()


class connectDevice:
    def collect(self):
        return 0
    def image_change(self):
        return 0
    def leapConnect(self):
        cwd = os.getcwd()
        data_recorder_path = cwd + r"\record_data.py"
        command = r"py -2 " + data_recorder_path
        print(command)
        global process
        process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE, text=True,bufsize=1)
        threading.Thread(target=lprint.leapPrint(self,process)).start()
        win.mainloop()

class lprint:
    def leapPrint(self,process):
        while process.poll() is None:
            output = process.stdout.readline()
            print(output)
            app.connect_text.insert(tk.END, output)
            app.connect_text.see(tk.END)
            win.update_idletasks()
            win.update()
            lprint.leapPrint(self,process)

            # force refresh of the widget to be sure that thing are displayed








if __name__ == '__main__':

    win = tk.Tk()
    win.title('leap controller')  # give a title name
    app = App(win)
    win.mainloop()