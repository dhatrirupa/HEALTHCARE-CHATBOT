from tkinter import *
from tkinter import ttk
import mysql.connector
from tkinter import messagebox
import time
import nltk
import random
import numpy as np
import json
import pickle
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer=WordNetLemmatizer()

with open('intents.json') as json_file:
    intents = json.load(json_file)

words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
model=load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
  sentence_words=nltk.word_tokenize(sentence)
  sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]
  return sentence_words

def bag_of_words(sentence):
  sentence_words=clean_up_sentence(sentence)
  bag=[0]*len(words)
  for w in sentence_words:
    for i,word in enumerate(words):
      if word == w:
        bag[i]=1
  return np.array(bag)

def predict_class(sentence):
  bow=bag_of_words(sentence)
  res=model.predict(np.array([bow]))[0]
  ERROR_THRESHOLD=0.25
  results=[[i,r] for i,r in enumerate(res) if r> ERROR_THRESHOLD]

  results.sort(key=lambda x:x[1],reverse=True)
  return_list=[]
  for r in results:
    return_list.append({'intent': classes[r[0]],'probability':str(r[1])})
  return return_list

def get_response(intents_list,intents_json):
  tag=intents_list[0]['intent']
  list_of_intents=intents_json['intents']
  for i in list_of_intents:
    if i['tag']==tag:
      result=random.choice(i['responses'])
      break
  return result

def chat(user_input):
    # Predict the intent of the user input
    intents_list = predict_class(user_input)
    # Get response based on the predicted intent
    response = get_response(intents_list, intents)
    return response

saved_username = ["You"]
window_size = "500x500"

class ChatInterface(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.tl_bg = "#EEEEEE"
        self.tl_bg2 = "#EEEEEE"
        self.tl_fg = "#000000"
        self.font = "Verdana 10"

        menu = Menu(self.master)
        self.master.config(menu=menu, bd=5)
        # Menu bar

        # File
        file = Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file)
        # file.add_command(label="Save Chat Log", command=self.save_chat)
        file.add_command(label="Clear Chat", command=self.clear_chat)
        #  file.add_separator()
        file.add_command(label="Exit", command=self.chatexit)

        # Options
        options = Menu(menu, tearoff=0)
        menu.add_cascade(label="Options", menu=options)

        # font
        font = Menu(options, tearoff=0)
        options.add_cascade(label="Font", menu=font)
        font.add_command(label="Default", command=self.font_change_default)
        font.add_command(label="Times", command=self.font_change_times)
        font.add_command(label="System", command=self.font_change_system)
        font.add_command(label="Helvetica", command=self.font_change_helvetica)
        font.add_command(label="Fixedsys", command=self.font_change_fixedsys)

        # color theme
        color_theme = Menu(options, tearoff=0)
        options.add_cascade(label="Color Theme", menu=color_theme)
        color_theme.add_command(label="Default", command=self.color_theme_default)
        # color_theme.add_command(label="Night",command=self.)
        color_theme.add_command(label="Grey", command=self.color_theme_grey)
        color_theme.add_command(label="Blue", command=self.color_theme_dark_blue)

        color_theme.add_command(label="Torque", command=self.color_theme_turquoise)
        color_theme.add_command(label="Hacker", command=self.color_theme_hacker)
        # color_theme.add_command(label='Mkbhd',command=self.MKBHD)

        help_option = Menu(menu, tearoff=0)
        menu.add_cascade(label="Help", menu=help_option)
        # help_option.add_command(label="Features", command=self.features_msg)
        help_option.add_command(label="About MedBot", command=self.msg)
        help_option.add_command(label="Developers", command=self.about)

        self.text_frame = Frame(self.master, bd=6)
        self.text_frame.pack(expand=True, fill=BOTH)

        # scrollbar for text box
        self.text_box_scrollbar = Scrollbar(self.text_frame, bd=0)
        self.text_box_scrollbar.pack(fill=Y, side=RIGHT)

        # contains messages
        self.text_box = Text(self.text_frame, yscrollcommand=self.text_box_scrollbar.set, state=DISABLED,
                             bd=1, padx=6, pady=6, spacing3=8, wrap=WORD, bg=None, font="Verdana 10", relief=GROOVE,
                             width=10, height=1)
        self.text_box.pack(expand=True, fill=BOTH)
        self.text_box_scrollbar.config(command=self.text_box.yview)

        # frame containing user entry field
        self.entry_frame = Frame(self.master, bd=1)
        self.entry_frame.pack(side=LEFT, fill=BOTH, expand=True)

        # entry field
        self.entry_field = Entry(self.entry_frame, bd=1, justify=LEFT)
        self.entry_field.pack(fill=X, padx=6, pady=6, ipady=3)
        # self.users_message = self.entry_field.get()

        # frame containing send button and emoji button
        self.send_button_frame = Frame(self.master, bd=0)
        self.send_button_frame.pack(fill=BOTH)

        # send button
        self.send_button = Button(self.send_button_frame, text="Send", width=5, relief=GROOVE, bg='white',
                                  bd=1, command=lambda: self.send_message_insert(), activebackground="#FFFFFF",
                                  activeforeground="#000000")
        self.send_button.pack(side=LEFT, ipady=8, expand=True)
        self.master.bind("<Return>", self.send_message_insert)
        self.last_sent_label(date="No messages sent.")
    

    def last_sent_label(self, date):

        try:
            self.sent_label.destroy()
        except AttributeError:
            pass

        self.sent_label = Label(self.entry_frame, font="Verdana 7", text=date, bg=self.tl_bg2, fg=self.tl_fg)
        self.sent_label.pack(side=LEFT, fill=BOTH, padx=3)

    def clear_chat(self):
        self.text_box.config(state=NORMAL)
        self.last_sent_label(date="No messages sent.")
        self.text_box.delete(1.0, END)
        self.text_box.delete(1.0, END)
        self.text_box.config(state=DISABLED)

    def chatexit(self):
        exit()

    def msg(self):
        tkinter.messagebox.showinfo("MedBot v1.0",
                                    'MedBot is a chatbot for answering health related queries\nIt is based on retrival-based NLP using pythons NLTK tool-kit module\nGUI is based on Tkinter\nIt can answer questions regarding users health status')

    def about(self):
        tkinter.messagebox.showinfo("MedBot Developers",
                                    "Dhatri,Ramya")

    def send_message_insert(self):
        user_input = self.entry_field.get()
        pr1 = "Human : " + user_input + "\n"
        self.text_box.configure(state=NORMAL)
        self.text_box.insert(END, pr1)
        self.text_box.configure(state=DISABLED)
        self.text_box.see(END)
        ob = chat(user_input)
        pr = "MedBot : " + ob + "\n"
        self.text_box.configure(state=NORMAL)
        self.text_box.insert(END, pr)
        self.text_box.configure(state=DISABLED)
        self.text_box.see(END)
        self.last_sent_label(str(time.strftime("Last message sent: " + '%B %d, %Y' + ' at ' + '%I:%M %p')))
        self.entry_field.delete(0, END)

    def font_change_default(self):
        self.text_box.config(font="Verdana 10")
        self.entry_field.config(font="Verdana 10")
        self.font = "Verdana 10"

    def font_change_times(self):
        self.text_box.config(font="Times")
        self.entry_field.config(font="Times")
        self.font = "Times"

    def font_change_system(self):
        self.text_box.config(font="System")
        self.entry_field.config(font="System")
        self.font = "System"

    def font_change_helvetica(self):
        self.text_box.config(font="helvetica 10")
        self.entry_field.config(font="helvetica 10")
        self.font = "helvetica 10"

    def font_change_fixedsys(self):
        self.text_box.config(font="fixedsys")
        self.entry_field.config(font="fixedsys")
        self.font = "fixedsys"

    def color_theme_default(self):
        self.master.config(bg="#EEEEEE")
        self.text_frame.config(bg="#EEEEEE")
        self.entry_frame.config(bg="#EEEEEE")
        self.text_box.config(bg="#FFFFFF", fg="#000000")
        self.entry_field.config(bg="#FFFFFF", fg="#000000", insertbackground="#000000")
        self.send_button_frame.config(bg="#EEEEEE")
        self.send_button.config(bg="#FFFFFF", fg="#000000", activebackground="#FFFFFF", activeforeground="#000000")
        self.sent_label.config(bg="#EEEEEE", fg="#000000")

        self.tl_bg = "#FFFFFF"
        self.tl_bg2 = "#EEEEEE"
        self.tl_fg = "#000000"

    # Dark
    def color_theme_dark(self):
        self.master.config(bg="#2a2b2d")
        self.text_frame.config(bg="#2a2b2d")
        self.text_box.config(bg="#212121", fg="#FFFFFF")
        self.entry_frame.config(bg="#2a2b2d")
        self.entry_field.config(bg="#212121", fg="#FFFFFF", insertbackground="#FFFFFF")
        self.send_button_frame.config(bg="#2a2b2d")
        self.send_button.config(bg="#212121", fg="#FFFFFF", activebackground="#212121", activeforeground="#FFFFFF")
        self.sent_label.config(bg="#2a2b2d", fg="#FFFFFF")

        self.tl_bg = "#212121"
        self.tl_bg2 = "#2a2b2d"
        self.tl_fg = "#FFFFFF"

    # Grey
    def color_theme_grey(self):
        self.master.config(bg="#444444")
        self.text_frame.config(bg="#444444")
        self.text_box.config(bg="#4f4f4f", fg="#ffffff")
        self.entry_frame.config(bg="#444444")
        self.entry_field.config(bg="#4f4f4f", fg="#ffffff", insertbackground="#ffffff")
        self.send_button_frame.config(bg="#444444")
        self.send_button.config(bg="#4f4f4f", fg="#ffffff", activebackground="#4f4f4f", activeforeground="#ffffff")
        self.sent_label.config(bg="#444444", fg="#ffffff")

        self.tl_bg = "#4f4f4f"
        self.tl_bg2 = "#444444"
        self.tl_fg = "#ffffff"

    def color_theme_turquoise(self):
        self.master.config(bg="#003333")
        self.text_frame.config(bg="#003333")
        self.text_box.config(bg="#669999", fg="#FFFFFF")
        self.entry_frame.config(bg="#003333")
        self.entry_field.config(bg="#669999", fg="#FFFFFF", insertbackground="#FFFFFF")
        self.send_button_frame.config(bg="#003333")
        self.send_button.config(bg="#669999", fg="#FFFFFF", activebackground="#669999", activeforeground="#FFFFFF")
        self.sent_label.config(bg="#003333", fg="#FFFFFF")

        self.tl_bg = "#669999"
        self.tl_bg2 = "#003333"
        self.tl_fg = "#FFFFFF"

        # Blue

    def color_theme_dark_blue(self):
        self.master.config(bg="#263b54")
        self.text_frame.config(bg="#263b54")
        self.text_box.config(bg="#1c2e44", fg="#FFFFFF")
        self.entry_frame.config(bg="#263b54")
        self.entry_field.config(bg="#1c2e44", fg="#FFFFFF", insertbackground="#FFFFFF")
        self.send_button_frame.config(bg="#263b54")
        self.send_button.config(bg="#1c2e44", fg="#FFFFFF", activebackground="#1c2e44", activeforeground="#FFFFFF")
        self.sent_label.config(bg="#263b54", fg="#FFFFFF")

        self.tl_bg = "#1c2e44"
        self.tl_bg2 = "#263b54"
        self.tl_fg = "#FFFFFF"

    # Hacker
    def color_theme_hacker(self):
        self.master.config(bg="#0F0F0F")
        self.text_frame.config(bg="#0F0F0F")
        self.entry_frame.config(bg="#0F0F0F")
        self.text_box.config(bg="#0F0F0F", fg="#33FF33")
        self.entry_field.config(bg="#0F0F0F", fg="#33FF33", insertbackground="#33FF33")
        self.send_button_frame.config(bg="#0F0F0F")
        self.send_button.config(bg="#0F0F0F", fg="#FFFFFF", activebackground="#0F0F0F", activeforeground="#FFFFFF")
        self.sent_label.config(bg="#0F0F0F", fg="#33FF33")

        self.tl_bg = "#0F0F0F"
        self.tl_bg2 = "#0F0F0F"
        self.tl_fg = "#33FF33"

    # Default font and color theme
    def default_format(self):
        self.font_change_default()
        self.color_theme_default()


def doctor():
    global window
    window = Tk()
    window.geometry('1800x1200') 
    window.title("Hospital Management System")
    frame = Frame(window, width=1600, height=54, bg="light blue")
    frame.place(x=0, y=0)
    lb = Label(window, text="MANAGE DOCTOR", bg="light blue", font=('calibre', 30, 'bold'))
    lb.place(x=615, y=0)

    lc = Label(window, text="NAME", font=('calibre', 20, 'bold'))
    lc.place(x=280, y=200)
    
    ld = Label(window, text="SPECIALIZATION", font=('calibre', 20, 'bold'))
    ld.place(x=280, y=250)

    le = Label(window, text="CONTACT", font=('calibre', 20, 'bold'))
    le.place(x=280, y=300)

    lf = Label(window, text="ADDRESS", font=('calibre', 20, 'bold'))
    lf.place(x=1200, y=200)

    global Name, specialization, contact, address
    Name = StringVar()
    specialization = StringVar()
    contact = StringVar()
    address = StringVar()
    global ea, eb, ec, ed
    ea = Entry(window, textvar=Name, width=20, font=('calibre', 18, 'bold'))
    ea.place(x=550, y=200)

    eb = Entry(window, textvar=specialization, width=20, font=('calibre', 18, 'bold'))
    eb.place(x=550, y=250)

    ec = Entry(window, textvar=contact, width=20, font=('calibre', 18, 'bold'))
    ec.place(x=550, y=300)

    ed = Entry(window, textvar=address, width=20, font=('calibre', 18, 'bold'))
    ed.place(x=900, y=200)

    ba = Button(window, text="BACK", bg="orange", font=('calibre', 18, 'bold'), width=15, command=back)
    ba.place(x=0, y=1)

    bb = Button(window, text="SUBMIT", bg="orange", font=('calibre', 15, 'bold'), width=15, bd=5, command=doctor1)
    bb.place(x=300, y=500)

    bc = Button(window, text="DELETE", bg="orange", font=('calibre', 15, 'bold'), width=15, bd=5, command=delete)
    bc.place(x=530, y=500)

    bd = Button(window, text="UPDATE", bg="orange", font=('calibre', 15, 'bold'), width=15, bd=5, command=update)
    bd.place(x=760, y=500)

    be = Button(window, text="VIEW", bg="orange", font=('calibre', 15, 'bold'), width=15, bd=5, command=show)
    be.place(x=990, y=500)

def patient():
    global window
    window = Tk()
    window.geometry('1800x1200') 
    window.title("Hospital Management System")
    frame = Frame(window, width=1600, height=54, bg="violet")
    frame.place(x=0, y=0)
    lb = Label(window, text="MANAGE PATIENT", bg="violet", font=('calibre', 30, 'bold'))
    lb.place(x=615, y=0)

    lg = Label(window, text="NAME", font=('calibre', 20, 'bold'))
    lg.place(x=280, y=200)
    
    lh = Label(window, text="GENDER", font=('calibre', 20, 'bold'))
    lh.place(x=280, y=250)

    li = Label(window, text="CONTACT", font=('calibre', 20, 'bold'))
    li.place(x=280, y=300)

    lj = Label(window, text="ADDRESS", font=('calibre', 20, 'bold'))
    lj.place(x=280, y=350)

    lk = Label(window, text="SEARCH", font=('calibre', 20, 'bold'))
    lk.place(x=1200, y=200)

    global Name, Gender, Contact, Address, Search
    Name = StringVar()
    Gender = StringVar()
    Contact = StringVar()
    Address = StringVar()
    Search = StringVar()
    global ef, eg, eh, ei, ej
   
    ef = Entry(window, textvar=Name, width=20, font=('calibre', 18, 'bold'))
    ef.place(x=550, y=200)

    eg = Entry(window, textvar=Gender, width=20, font=('calibre', 18, 'bold'))
    eg.place(x=550, y=250)

    eh = Entry(window, textvar=Contact, width=20, font=('calibre', 18, 'bold'))
    eh.place(x=550, y=300)

    ei = Entry(window, textvar=Address, width=20, font=('calibre', 18, 'bold'))
    ei.place(x=550, y=350)

    ej = Entry(window, textvar=Search, width=20, font=('calibre', 18, 'bold'))
    ej.place(x=900, y=200)

    bf = Button(window, text="BACK", bg="orange", font=('calibre', 18, 'bold'), width=15, command=back)
    bf.place(x=0, y=1)

    bg = Button(window, text="SUBMIT", bg="orange", font=('calibre', 15, 'bold'), width=15, bd=5, command=patient1)
    bg.place(x=300, y=500)

    bh = Button(window, text="DELETE", bg="orange", font=('calibre', 15, 'bold'), width=15, bd=5, command=delete)
    bh.place(x=530, y=500)

    bi = Button(window, text="SEARCH", bg="orange", font=('calibre', 15, 'bold'), width=15, bd=5, command=search)
    bi.place(x=760, y=500)

    bj = Button(window, text="VIEW", bg="orange", font=('calibre', 15, 'bold'), width=15, bd=5, command=show)
    bj.place(x=990, y=500)

def appointment():
    global window
    window = Tk()
    window.geometry('1800x1200') 
    window.title("Hospital Management System")
    frame = Frame(window, width=1600, height=54, bg="violet")
    frame.place(x=0, y=0)
    lb = Label(window, text="MANAGE APPOINTMENT", bg="violet", font=('calibre', 30, 'bold'))
    lb.place(x=615, y=0)

    lg = Label(window, text="DOCTOR NAME", font=('calibre', 20, 'bold'))
    lg.place(x=280, y=200)
    
    lh = Label(window, text="PATIENT NAME", font=('calibre', 20, 'bold'))
    lh.place(x=280, y=250)

    li = Label(window, text="DATE", font=('calibre', 20, 'bold'))
    li.place(x=280, y=300)

    lj = Label(window, text="TIME", font=('calibre', 20, 'bold'))
    lj.place(x=280, y=350)

    lk = Label(window, text="SEARCH", font=('calibre', 20, 'bold'))
    lk.place(x=1200, y=200)

    global e1, e2, e3, ei, ek
    global search, doctor, patient, date, time
    doctor = StringVar()
    patient = StringVar()
    date = StringVar()
    time = StringVar()
    search = StringVar()
    e1 = Entry(window, textvar=doctor, width=20, font=('calibre', 18, 'bold'))
    e1.place(x=550, y=200)

    e2 = Entry(window, textvar=patient, width=20, font=('calibre', 18, 'bold'))
    e2.place(x=550, y=250)

    e3 = Entry(window, textvar=date, width=20, font=('calibre', 18, 'bold'))
    e3.place(x=550, y=300)

    ei = Entry(window, textvar=time, width=20, font=('calibre', 18, 'bold'))
    ei.place(x=550, y=350)

    ek = Entry(window, textvar=search, width=20, font=('calibre', 18, 'bold'))
    ek.place(x=900, y=200)

    bf = Button(window, text="BACK", bg="orange", font=('calibre', 18, 'bold'), width=15, command=back)
    bf.place(x=0, y=1)

    bg = Button(window, text="SUBMIT", bg="orange", font=('calibre', 15, 'bold'), width=15, bd=5, command=appointment1)
    bg.place(x=300, y=500)

    bh = Button(window, text="DELETE", bg="orange", font=('calibre', 15, 'bold'), width=15, bd=5, command=delete)
    bh.place(x=530, y=500)

    bi = Button(window, text="SEARCH", bg="orange", font=('calibre', 15, 'bold'), width=15, bd=5, command=bitch)
    bi.place(x=760, y=500)

    bj = Button(window, text="VIEW", bg="orange", font=('calibre', 15, 'bold'), width=15, bd=5, command=show)
    bj.place(x=990, y=500)

def chat_interface():
    root = Tk()
    a = ChatInterface(root)
    root.geometry(window_size)
    root.title("MedBot")
    root.iconbitmap('MedBot.jpg')
    root.mainloop()


def back():
    window.destroy()

def doctor1():
    a = ea.get()
    b = eb.get()
    c = ec.get()
    d = ed.get()
    conn = mysql.connector.connect(host="localhost", user="root", password="mysql", db="Hospital")
    cs = conn.cursor()
    sql = "INSERT INTO doctor(name,specialization,contact,address)VALUES(%s,%s,%s,%s)"
    val = (a, b, c, d)
    cs.execute(sql, val)
    conn.commit()
    cs.close()
    conn.close()
    ea.delete(first=0, last=20)
    eb.delete(first=0, last=20)
    ec.delete(first=0, last=20)
    ed.delete(first=0, last=20)

def show():
    window6 = Tk()
    window6.geometry('1800x1200')
    tree = ttk.Treeview(window6, column=(1, 2, 3, 4), show="headings", height=30)
    tree.heading(1, text="Name")
    tree.heading(2, text="Gender")
    tree.heading(3, text="Contact")
    tree.heading(4, text="Address")
    tree.place(x=400, y=100)
    frame2 = Frame(window6, width=1600, height=54, bg="violet")
    frame2.place(x=0, y=0)
    lb = Label(window6, text="List Of Patient", bg="violet", font=('calibre', 30, 'bold'))
    lb.place(x=615, y=0)

    conn = mysql.connector.connect(host="localhost", user="root", password="mysql", db="Hospital")
    cs = conn.cursor()
    cs.execute("SELECT * FROM patient")
    rows = cs.fetchall()
    for x in rows:
        tree.insert('', 'end', values=x)

def delete():
    messagebox.askquestion("Confirmation", "Are you sure want to delete?")
    b = ef.get()
    conn = mysql.connector.connect(host="localhost", user="root", password="mysql", db="Hospital")
    cs = conn.cursor()
    sql = "DELETE FROM patient WHERE Name=%s"
    val = (b,)
    cs.execute(sql, val)     
    conn.commit()
    cs.close()
    conn.close()

def search():
    s = ej.get()
    conn = mysql.connector.connect(host="localhost", user="root", password="mysql", db="Hospital")
    cs = conn.cursor()
    sql = "SELECT * FROM patient WHERE Name=%s "
    val = (s,)
    cs.execute(sql, val)
    rows = cs.fetchall()
    for x in rows:
        a = x[0]
        b = x[1]
        c = x[2]
        d = x[3]
    if s == a:
        ef.insert(0, a)
        eg.insert(0, b)
        eh.insert(0, c)
        ei.insert(0, d)

def patient1():
    a = ef.get()
    b = eg.get()
    c = eh.get()
    d = ei.get()
    conn = mysql.connector.connect(host="localhost", user="root", password="mysql", db="Hospital")
    cs = conn.cursor()
    sql = "INSERT INTO patient(Name,gender,contact,address)VALUES(%s,%s,%s,%s)"
    val = (a, b, c, d)
    cs.execute(sql, val)
    conn.commit()
    cs.close()
    conn.close()
    ef.delete(first=0, last=20)
    eg.delete(first=0, last=20)
    eh.delete(first=0, last=20)
    ei.delete(first=0, last=20)

def appointment1():
    a = e1.get()
    b = e2.get()
    c = e3.get()
    d = ei.get()
    conn = mysql.connector.connect(host="localhost", user="root", password="mysql", db="Hospital")
    cs = conn.cursor()
    sql = "INSERT INTO appointment(doctor,patient,date,time)VALUES(%s,%s,%s,%s)"
    val = (a, b, c, d)
    cs.execute(sql, val)
    conn.commit()
    cs.close()
    conn.close()
    e1.delete(first=0, last=20)
    e2.delete(first=0, last=20)
    e3.delete(first=0, last=20)
    ei.delete(first=0, last=20)

def merge_all():
    root = Tk()
    root.geometry('600x400')
    root.title("Hospital Management System")

    button_doctor = Button(root, text="Doctor", command=doctor)
    button_doctor.pack(pady=10)

    button_patient = Button(root, text="Patient", command=patient)
    button_patient.pack(pady=10)

    button_appointment = Button(root, text="Appointment", command=appointment)
    button_appointment.pack(pady=10)

    button_chat_bot = Button(root, text="Chat with Chatbot", command=chat_interface)
    button_chat_bot.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    merge_all()
