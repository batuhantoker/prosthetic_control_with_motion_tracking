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

thread_pool_executor = futures.ThreadPoolExecutor(max_workers=2)
win = tk.Tk()
win.title('leap controller')  # give a title name
class App:
# create labels
# name label
name_label = ttk.Label(win, text="Name : ")
name_label.grid(row=0, column=0, sticky=tk.W)


# age label
age_label = ttk.Label(win, text="Age : ")
age_label.grid(row=1, column=0, sticky=tk.W)


# gender label
gender_label = ttk.Label(win, text="Gender : ")
gender_label.grid(row=2, column=0, sticky=tk.W)

# Create entry box
# name entry box
name_var = tk.StringVar()
name_entrybox = ttk.Entry(win, width=16, textvariable=name_var)
name_entrybox.grid(row=0, column=1)
name_entrybox.focus()


# age entry box
age_var = tk.StringVar()
age_entrybox = ttk.Entry(win, width=16, textvariable=age_var)
age_entrybox.grid(row=1, column=1)

# gender entry box
# create combobox
gender_var = tk.StringVar()
gender_combobox = ttk.Combobox(win, width=13, textvariable=gender_var, state="readonly")
gender_combobox['values'] = ('Male', 'Female', 'Other')
gender_combobox.current(0)
gender_combobox.grid(row=2, column=1)



# create check button
checkbtn_var = tk.IntVar()
checkbtn = ttk.Checkbutton(win, text='Sensor is calibrated', variable=checkbtn_var)
checkbtn.grid(row=6, columnspan=3)


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

def bar(duration):
    progress['value'] = 20
    win.update_idletasks()
    time.sleep(duration/4)
    progress['value'] = 50
    win.update_idletasks()
    time.sleep(duration/4)
    progress['value'] = 80
    win.update_idletasks()
    time.sleep(duration/4)
    progress['value'] = 100


def collect():
    return 0
def image_change():
    return 0
def leapConnect():
    cwd = os.getcwd()
    data_recorder_path = cwd + r"\record_data.py"
    command = r"py -2 " + data_recorder_path
    print(command)
    global process
    process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE, text=True)
    process.poll()
    for line in process.stdout:
        # the real code does filtering here
        connect_text.insert(tk.END, line)
        connect_text.see(tk.END)
        # force refresh of the widget to be sure that thing are displayed
        win.update_idletasks()
        win.update()

def start_leap_in_bg():
    loop_thread = threading.Thread(target=leapConnect())
    loop_thread.daemon = True  #
    loop_thread.start()

connect_text = tk.Text(win, relief="sunken")
connect_text.grid(row=16, column=0)



progress = Progressbar(win, orient=HORIZONTAL, length=100, mode='determinate')
progress.grid(row=27, column=0)


theButton= Button(win, text='Start', command =lambda: start_leap_in_bg()).grid(row=15, column=0)
Button(win, text='Connect', command = lambda:[collect(),bar(3),image_change]).grid(row=15, column=50)
# Get the current working directory
cwd = os.getcwd()
path = cwd + "\\pinch.png"
image = tk.PhotoImage(file="pinch.png") ## Replace with gif
label = tk.Label(image=image)
label.grid(row=100, column=100)

# train button
train_button = ttk.Button(win, text="Train", command=action)
train_button.grid(row=100, column=0)

# test button
test_button = ttk.Button(win, text="Test", command=action)
test_button.grid(row=100, column=50)


# classifier label
classifier_label = ttk.Label(win, text="Classifier")
classifier_label.grid(row=100, column=5, sticky=tk.W)

win.update_idletasks()
win.update()
win.mainloop()  # application is not closed auto