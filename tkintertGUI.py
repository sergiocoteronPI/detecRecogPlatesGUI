
import tkinter
from tkinter import *
import tkinter.filedialog
from tkinter.filedialog import askopenfilename

from PIL import Image, ImageTk

import cv2
import os

import threading

import tensorflow as tf

import matDetec
import matRecog

class appMatriculas:

    #Función que se inicia en cuanto arranca el programa.
    def __init__(self, window):
        
        #Titulo de la ventana de la aplicación.
        window.title("APP - Detección y reconocimiento de matrículas")

        #Tamaño de la pantala en la que estamos trabajando y tamaños de referencia para que los botones mantengan su tamaño fijo.
        self.referenciaWidth = 1280
        self.referenciaHeight = 720

        self.screen_width = window.winfo_screenwidth()
        self.screen_height = window.winfo_screenheight()
        window.geometry(str(self.screen_width) + 'x' + str(self.screen_height))

        #Con esta función pretendo hacer la página responsive.
        self.responsive()

        #Iniciamos los botones les asignamos valor y posición.
        self.iniciarBotones()

        #Variables de control
        self.camControl = False
        self.controlOCR = False
        self.controlNMS = True

        self.controlDetection = False
        self.controlOCR = False
        self.controlEscribirTxt = False

        self.contadorImagen = 0

        #Cargamos las clases para analizar las imágenes#
        #***************************************************************************#
        self.claseMatDetec = matDetec.clasMatDetec
        self.matDetecClass = matDetec.matDetec()

        self.claseMatOCR = matRecog.clasMatOcr
        self.matOCRClass = matRecog.matRecog()
        #***************************************************************************#


    def responsive(self):
        
        for i in range(11):
            window.grid_rowconfigure(i, weight=1)
            window.grid_columnconfigure(i, weight=1) 


    def iniciarBotones(self):

        self._width = int(self.screen_width*15/self.referenciaWidth)
        self._widthCol2 = 1#int(self.screen_width*1/self.referenciaWidth)

        self._height = int(self.screen_height*3/self.referenciaHeight)

        # ================================================ COLUMNA 0 ================================================ #

        self.btnImagen = Button(window, text="Imagen", font=("Arial Bold", 12), command = self.abrirImagen, height = self._height, width = self._width)
        self.btnImagen.grid(column=0, row=0)

        self.btnImagenes = Button(window, text="Imagenes", font=("Arial Bold", 12), command = self.abrirDirectorio, height = self._height, width = self._width)
        self.btnImagenes.grid(column=0, row=1)

        self.btnCam = Button(window, text="Cam", font=("Arial Bold", 12), command = self.webcam_control, height = self._height, width = self._width)
        self.btnCam.grid(column=0, row=2)

        self.btnVideo = Button(window, text="Video", font=("Arial Bold", 12), command = self.abrirVideo, height = self._height, width = self._width)
        self.btnVideo.grid(column=0, row=3)

        #############################################################################################################################################
        window.grid_rowconfigure(4, minsize=20)
        #############################################################################################################################################

        self.btnDetectar = Button(window, text="Deteccion", font=("Arial Bold", 12), command = self.activarDeteccion,
                                  height = self._height, width = self._width)
        self.btnDetectar.grid(column=0, row=5)

        self.btnNMS = Button(window, text="NMS", font=("Arial Bold", 12), command = self.activarNMS, height = self._height, width = self._width)
        self.btnNMS.grid(column=0, row=6)

        #############################################################################################################################################
        window.grid_rowconfigure(7, minsize=20)
        #############################################################################################################################################

        self.btnOCR = Button(window, text="OCR", font=("Arial Bold", 12), command = self.activarOCR, height = self._height, width = self._width)
        self.btnOCR.grid(column=0, row=8)

        #############################################################################################################################################
        window.grid_rowconfigure(9, minsize=20)
        #############################################################################################################################################

        self.btnEscribirTxt = Button(window, text="Escribir txt", font=("Arial Bold", 12), command = self.activarEscribirTxt, height = self._height, width = self._width)
        self.btnEscribirTxt.grid(column=0, row=10)

        # ================================================ COLUMNA 0 ================================================ #


        # ================================================ COLUMNA 1 ================================================ #

        self.controlColorDetectar = Label(window, text="", height = self._height, width = self._widthCol2, bg="red")
        self.controlColorDetectar.grid(column=1, row=5, padx=(5, 5))

        self.controlColorNMS = Label(window, text="", height = self._height, width = self._widthCol2, bg="green")
        self.controlColorNMS.grid(column=1, row=6, padx=(5, 5))

        self.controlColorOCR = Label(window, text="", height = self._height, width = self._widthCol2, bg="red")
        self.controlColorOCR.grid(column=1, row=8, padx=(5, 5))

        self.ControlColorEscribirTxt = Label(window, text="", height = self._height, width = self._widthCol2, bg="red")
        self.ControlColorEscribirTxt.grid(column=1, row=10, padx=(5, 5))

        # ================================================ COLUMNA 1 ================================================ #

        self.panel = Label(window, text="Detecciones", height = 30, width = 100, borderwidth=2, relief="solid", anchor=NW)
        self.panel.grid(row=1, column=2, columnspan=10, rowspan=10, padx=(20, 20))

        self.cuadroDeTexto("")
        self.resetCanvas()

        self.panelOCR = Label(window, text="OCR", height = 30, width = 30, borderwidth=2, relief="solid", anchor=N)
        self.panelOCR.grid(row=1, column=13, columnspan=2, rowspan=10, padx=(20, 20))

        # ================================================ BOTONES SUPERIORES ================================================ #

        #############################################################################################################################################
        window.grid_columnconfigure(2, minsize=50)
        #############################################################################################################################################

        self.btnRetrocederTodo = Button(window, text="<<", font=("Arial Bold", 15), command = self.primeraImagen, height = 1, width = 5)
        self.btnRetrocederTodo.grid(column=3, row=0)

        self.btnRetrocederUno = Button(window, text="<", font=("Arial Bold", 15), command = self.anteriorImagen, height = 1, width = 5)
        self.btnRetrocederUno.grid(column=4, row=0)

        self.btnAvanzarUna = Button(window, text=">", font=("Arial Bold", 15), command = self.siguienteImagen, height = 1, width = 5)
        self.btnAvanzarUna.grid(column=5, row=0)

        self.btnAvanzarTodo = Button(window, text=">>", font=("Arial Bold", 15), command = self.ultimaImagen, height = 1, width = 5)
        self.btnAvanzarTodo.grid(column=6, row=0)
    

    def resetCanvas(self):

        self.canvasDetection = tkinter.Canvas(window, width=400, height=400)#, background='white'
        self.canvasDetection.grid(row=1, column=2, columnspan=10, rowspan=10, padx=(20, 20))


    def crearCanvasOCR(self, _row, recortMatricula, predictionOCR):

        if _row == 2:
            
            self.canvasOCR_2 = tkinter.Canvas(window, width=150, height=90)
            self.canvasOCR_2.grid(row=_row, column=13, columnspan=2)
            
            self.imageOCR_2 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cv2.resize(recortMatricula, (150,64)), cv2.COLOR_BGR2RGB)))
            self.canvasOCR_2.create_image(0, 0, image=self.imageOCR_2, anchor=NW)

            self.canvasOCR_2.create_text(75, 70, font=("Times New Roman", 15, "bold"), text=predictionOCR)

        if _row == 4:
            
            self.canvasOCR_4 = tkinter.Canvas(window, width=150, height=90)
            self.canvasOCR_4.grid(row=_row, column=13, columnspan=2, rowspan=1, padx=(10, 10))

            self.imageOCR_4 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cv2.resize(recortMatricula, (150,64)), cv2.COLOR_BGR2RGB)))
            self.canvasOCR_4.create_image(0, 0, image=self.imageOCR_4, anchor=NW)

            self.canvasOCR_4.create_text(75, 70, font=("Times New Roman", 15, "bold"), text=predictionOCR)

        if _row == 6:
            
            self.canvasOCR_6 = tkinter.Canvas(window, width=150, height=90)
            self.canvasOCR_6.grid(row=_row, column=13, columnspan=2, rowspan=1)

            self.imageOCR_6 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cv2.resize(recortMatricula, (150,64)), cv2.COLOR_BGR2RGB)))
            self.canvasOCR_6.create_image(0, 0, image=self.imageOCR_6, anchor=NW)

            self.canvasOCR_6.create_text(75, 70, font=("Times New Roman", 15, "bold"), text=predictionOCR)

        if _row == 10:
            
            self.canvasOCR_8 = tkinter.Canvas(window, width=150, height=90)
            self.canvasOCR_8.grid(row=_row, column=13, columnspan=2, rowspan=1)

            self.imageOCR_8 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cv2.resize(recortMatricula, (150,64)), cv2.COLOR_BGR2RGB)))
            self.canvasOCR_8.create_image(0, 0, image=self.imageOCR_8, anchor=NW)

            self.canvasOCR_8.create_text(75, 80, font=("Times New Roman", 15, "bold"), text=predictionOCR)


    def cuadroDeTexto(self, texto):

        self.textoInformacion = Label(window, text=texto, height = 2, width = 100, borderwidth=2, relief="solid", anchor=NW)
        self.textoInformacion.grid(row=10, column=2, columnspan=10, padx=(3, 3))


    def webcam_control(self):

        if(not self.camControl):

            self.camControl = True

            try:
                self.cap = cv2.VideoCapture(0)
            except:
                try:
                    self.cap = cv2.VideoCapture(-1)
                except:
                    try:
                        self.cap = cv2.VideoCapture(1)
                    except:
                        return

            self.stopEvent = threading.Event()
            self.thread = threading.Thread(target=self.iniciar_webcam)
            self.thread.start()

        else:

            self.resetCanvas()

            self.cap.release()
            self.stopEvent.set()

            self.camControl = False


    def iniciar_webcam(self):
        
        redimWidth, redimHeight = self.canvasDetection.winfo_width(),self.canvasDetection.winfo_height()

        while not self.stopEvent.is_set():

            try:
                _, frame = self.cap.read()
                frame = cv2.resize(frame, (redimWidth,redimHeight))
 
                image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvasDetection.create_image(0, 0, image=image, anchor=NW)
            except:
                self.cap.release()
                self.stopEvent.set()
                break
            
            try:
                cv2.waitKey(50)
            except:
                break


    def mostrarImagen(self):
        
        redimWidth, redimHeight = self.canvasDetection.winfo_width(),self.canvasDetection.winfo_height()
        try:
            #Al parecer los self.frame, self.image son super importantes asi que no eliminar
            self.frame = cv2.imread(self.filesNomb[self.contadorImagen])
            self.cuadroDeTexto("Imagen: " + str(self.contadorImagen+1) + "/" + str(len(self.filesNomb)) )

            if self.controlDetection:
                #codigo de deteccion
                self.imgOutDetec, self.boxDetec = self.matDetecClass.matDetecFunction(self.frame)

                if self.controlOCR:

                    #codigo OCR
                    _row = 2
                    for matriculasDetectadas in self.boxDetec:
                        
                        coordX, coordY, coordW, coordH = matriculasDetectadas["topleft"]["x"],matriculasDetectadas["topleft"]["y"],\
                                                         matriculasDetectadas["bottomright"]["x"],matriculasDetectadas["bottomright"]["y"]

                        recortMatricula = self.frame[coordY:coordH, coordX:coordW, :]

                        #cv2.imshow('sdf', recortMatricula)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()

                        predictionOCR = self.matOCRClass.matOCRFunction(cv2.cvtColor(recortMatricula, cv2.COLOR_BGR2GRAY))

                        # Mostramos los recortes de matrículas y la predicción hecha #
                        # ========================================================== #

                        if _row < 10:
                            self.crearCanvasOCR(_row, recortMatricula, predictionOCR)

                        _row += 2

                        # ========================================================== #

                self.frame = cv2.resize(self.imgOutDetec, (redimWidth,redimHeight))

            else:
                self.frame = cv2.resize(self.frame, (redimWidth,redimHeight))
  
            self.image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)))
            self.canvasDetection.create_image(0, 0, image=self.image, anchor=NW)
        except:
            self.cuadroDeTexto("Imagen: " + str(self.contadorImagen+1) + "/" + str(len(self.filesNomb)) + "\t Error: imposible mostrar imagen - " + self.filesNomb[self.contadorImagen])
            self.resetCanvas()
            return


    def siguienteImagen(self):

        try:
            if self.contadorImagen < len(self.filesNomb)-1:
                self.contadorImagen += 1
                self.mostrarImagen()
        except:
            return

    
    def anteriorImagen(self):
        
        try:
            if self.contadorImagen > 0 and len(self.filesNomb)>0:
                self.contadorImagen -= 1
                self.mostrarImagen()
        except:
            return

    def primeraImagen(self):
        
        try:
            if len(self.filesNomb)>0 and self.contadorImagen != 0:
                self.contadorImagen = 0
                self.mostrarImagen()
        except:
            return

    
    def ultimaImagen(self):
        
        try:
            if len(self.filesNomb)>0 and self.contadorImagen < len(self.filesNomb)-1:
                self.contadorImagen = len(self.filesNomb)-1
                self.mostrarImagen()
        except:
            return

    
    def filesRead(self, _path, archivosPermitidos):
        filesNomb = []
        for ruta, _, ficheros in os.walk(_path):
            for nombreFichero in ficheros:
                rutComp = os.path.join(ruta, nombreFichero)
                for arcPerm in archivosPermitidos:
                    if rutComp.endswith(arcPerm):
                        filesNomb.append(rutComp)
        return filesNomb

    def fileRead(self, _path, archivosPermitidos):
        filesNomb = []
        for arcPerm in archivosPermitidos:
            if _path.endswith(arcPerm):
                filesNomb.append(_path)
        return filesNomb

    def abrirImagen(self):

        _path = askopenfilename(initialdir = "/",title = "Elije un archivo",filetypes = (("jpg files","*.jpg"),
                                                                                            ("png files","*.png"),
                                                                                            ("jpeg files","*.jpeg"),
                                                                                            ("JPG files","*.JPG"),
                                                                                            ("all files","*.*")))

        self.filesNomb = self.fileRead(_path, ["jpg","jpeg","png","JPG"])
        
        if self.filesNomb != []:
            self.contadorImagen = 0
            self.mostrarImagen()

    
    def abrirDirectorio(self):
    
        _path = tkinter.filedialog.askdirectory(initialdir = "/",title = "Elije un directorio")
        self.filesNomb = self.filesRead(_path, ["jpg","jpeg","png","JPG"])

        if self.filesNomb != []:
            self.contadorImagen = 0
            self.mostrarImagen()

    
    def abrirVideo(self):
    
        _path = askopenfilename(initialdir = "/",title = "Elije un archivo",filetypes = (("mp4 files","*.mp4"),
                                                                                            ("avi files","*.avi"),
                                                                                            ("mpg files","*.mpg"),
                                                                                            ("all files","*.*")))

        self.filesNomb = self.fileRead(_path, ["mp4","avi","mpg"])                                                                                             


    def activarDeteccion(self):
        
        if(self.controlDetection):

            self.controlDetection = False

            self.controlColorDetectar = Label(window, text="", height = self._height, width = self._widthCol2, bg="red")
            self.controlColorDetectar.grid(column=1, row=5)
        else:

            self.controlDetection = True

            self.controlColorDetectar = Label(window, text="", height = self._height, width = self._widthCol2, bg="green")
            self.controlColorDetectar.grid(column=1, row=5)

    
    def activarNMS(self):
        
        if(self.controlNMS):

            self.controlNMS = False
            self.claseMatDetec.nms = self.controlNMS

            self.controlColorNMS = Label(window, text="", height = self._height, width = self._widthCol2, bg="red")
            self.controlColorNMS.grid(column=1, row=6)
        else:

            self.controlNMS = True
            self.claseMatDetec.nms = self.controlNMS

            self.controlColorNMS = Label(window, text="", height = self._height, width = self._widthCol2, bg="green")
            self.controlColorNMS.grid(column=1, row=6)

            
    def activarOCR(self):
        
        if(self.controlOCR):

            self.controlOCR = False

            self.controlColorOCR = Label(window, text="", height = self._height, width = self._widthCol2, bg="red")
            self.controlColorOCR.grid(column=1, row=8)
        else:

            self.controlOCR = True

            self.controlColorOCR = Label(window, text="", height = self._height, width = self._widthCol2, bg="green")
            self.controlColorOCR.grid(column=1, row=8)

    
    def activarEscribirTxt(self):

        if(self.controlEscribirTxt):

            self.controlEscribirTxt = False

            self.ControlColorEscribirTxt = Label(window, text="", height = self._height, width = self._widthCol2, bg="red")
            self.ControlColorEscribirTxt.grid(column=1, row=10)
        else:

            self.controlEscribirTxt = True

            self.ControlColorEscribirTxt = Label(window, text="", height = self._height, width = self._widthCol2, bg="green")
            self.ControlColorEscribirTxt.grid(column=1, row=10)


    def exit_window(self):

        try:
            self.cap.release()
        except:
            pass
    
        try:
            self.stopEvent.set()
        except:
            pass
            
        window.destroy()

    

if __name__ == "__main__":

    window = Tk()
    appMatriculas(window)
    window.mainloop()