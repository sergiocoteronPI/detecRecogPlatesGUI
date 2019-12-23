
import tkinter
from tkinter import *

import cv2
from PIL import Image
from PIL import ImageTk

import threading
import sys

class appMatriculas:

    #Función que se inicia en cuanto arranca el programa.
    def __init__(self, window):

        window.title("APP - Detección y reconocimiento de matrículas")

        #Tamaño de la pantala en la que estamos trabajando para que las dimensiones de
        self.referenciaWidth = 1280
        self.referenciaHeight = 720

        self.screen_width = window.winfo_screenwidth()
        self.screen_height = window.winfo_screenheight()
        window.geometry(str(self.screen_width) + 'x' + str(self.screen_height))

        self.responsive()

        self.btnImagen = None
        self.btnImagenes = None
        self.btnCam = None
        self.btnVideo = None

        self.iniciarBotones() 

        btn = Button(window, text="Salir", command = self.exit_window)
        btn.grid(column=1, row=0)

        btn = Button(window, text="Iniciar webcam", command = self.iniciar_webcam)
        btn.grid(column=2, row=0)

        lbl = Label(window, text="Programa para tratar de poder mostrar mis resultados", font=("Arial Bold", 12))
        lbl.grid(column=0, row=0)

        #self.cap = cv2.VideoCapture(0)

        self.panel = None

        #self.stopEvent = threading.Event()
        #self.thread = threading.Thread(target=self.iniciar_webcam)
        #self.thread.start()

    def responsive(self):
        
        for i in range(7):
            window.grid_rowconfigure(i, weight=1)
            window.grid_columnconfigure(i, weight=1) 

    def iniciarBotones(self):

        _width = int(self.screen_width*15/self.referenciaWidth)
        _height = int(self.screen_height*2/self.referenciaHeight)

        self.btnImagen = Button(window, text="Imagen", font=("Arial Bold", 12), command = self.exit_window, height = _height, width = _width)
        self.btnImagen.grid(column=0, row=3)

        self.btnImagenes = Button(window, text="Imagenes", font=("Arial Bold", 12), command = self.exit_window, height = _height, width = _width)
        self.btnImagenes.grid(column=0, row=4)

        self.btnCam = Button(window, text="Cam", font=("Arial Bold", 12), command = self.exit_window, height = _height, width = _width)
        self.btnCam.grid(column=0, row=5)

        self.btnVideo = Button(window, text="Video", font=("Arial Bold", 12), command = self.exit_window, height = _height, width = _width)
        self.btnVideo.grid(column=0, row=6)


    def iniciar_webcam(self):

        while not self.stopEvent.is_set():
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, (480,480))

            image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            if self.panel is None:
                self.panel = tkinter.Label(image=image)
                self.panel.image = image
                self.panel.grid(column=2, row=2)
            else:
                self.panel.configure(image=image)
                self.panel.image = image

        cap.release()
        cv2.destroyAllWindows()

    def exit_window(self):

        window.destroy()
        sys.exit

    

if __name__ == "__main__":

    window = Tk()
    my_gui = appMatriculas(window)
    window.mainloop()