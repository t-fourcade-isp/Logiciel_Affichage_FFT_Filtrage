# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:59:26 2019

@author: t.fourcade

Logiciel pour l'affichage, le calcul de FFT et le filtrage
"""

import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg
from scipy.signal import butter, sosfilt


if not QtWidgets.QApplication.instance():
    app = QtWidgets.QApplication()
else:
    app = QtWidgets.QApplication.instance()


pg.setConfigOption('background', 'w')


class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        """Initialisation de la fenêtre principale"""
        self.resize(1200, 800)
        self.setWindowTitle("Traitement des données")
        # Initilisation des données
        self.data = None
        # Ajout d'un widget central
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)
        # Ajout d'un layout
        mainLayout = QtWidgets.QGridLayout()
        centralWidget.setLayout(mainLayout)
        # Creation des boutons et labels
        self.nomFichier = QtWidgets.QLabel()
        btn1 = QtWidgets.QPushButton("Ouvrir")
        btn2 = QtWidgets.QPushButton("Fermer")
        btn3 = QtWidgets.QPushButton("Filtrer")
        btn4 = QtWidgets.QPushButton("Supprimer\nfiltre")
        btn5 = QtWidgets.QPushButton("Filtre de sélection")
        btn6 = QtWidgets.QPushButton("Cacher filtre de sélection")
        btn7 = QtWidgets.QPushButton("Sélectionner données")
        btn8 = QtWidgets.QPushButton("Annuler sélection")
        btn9 = QtWidgets.QPushButton("Utiliser données filtrées")
        self.fd = QtWidgets.QFileDialog(filter='*.txt')
        self.lbl1 = QtWidgets.QLabel("Fréquence d'échantillonage = {} Hz".format(0))
        self.lbl2 = QtWidgets.QLabel("Frequence de résonance = {} Hz".format(0))
        lbl3 = QtWidgets.QLabel("Ordre du filtre")
        self.cb1 = QtWidgets.QComboBox()
        self.cb1.addItems(["2", "3", "4", "5", "6", "7", "8", "9", "10"])
        self.cb1.setCurrentText("8")
        lbl4 = QtWidgets.QLabel("Frequence de coupure basse (Hz)")
        self.sb1 = QtWidgets.QSpinBox(minimum=1)
        self.sb1.setValue(40)
        lbl5 = QtWidgets.QLabel("Frequence de coupure haute (Hz)")
        self.sb2 = QtWidgets.QSpinBox(minimum=1)
        self.sb2.setValue(400)
        self.lbl6 = QtWidgets.QLabel("Type de filtre")
        self.cb2 = QtWidgets.QComboBox()
        self.cb2.addItems(["lowpass", "highpass", "bandpass", "bandstop"])
        # Creation du graph d'affichage de la courbe temps-déplacement
        self.pw1 = pg.PlotWidget(title="Temps-Déplacement")
        self.pw1.getPlotItem().showGrid(x=True, y=True)
        self.pw1.getPlotItem().getAxis('bottom').setPen('k')
        self.pw1.getPlotItem().getAxis('left').setPen('k')
        self.pw1.getPlotItem().setLabel('bottom', "Temps", "s")
        self.pw1.getPlotItem().setLabel('left', "Déplacemen", "nm")
        self.unfilteredPlot = self.pw1.getPlotItem().plot()
        self.filteredPlot = self.pw1.getPlotItem().plot()
        self.filteredPlot.setPen("g")
        self.unfilteredPlot.setPen("b")
        self.lr1 = pg.LinearRegionItem()
        # Creation du graph d'affichage de la courbe de FFT
        self.pw2 = pg.PlotWidget(title="FFT")
        self.pw2.getPlotItem().showGrid(x=True, y=True)
        self.pw2.getPlotItem().getAxis('bottom').setPen('k')
        self.pw2.getPlotItem().getAxis('left').setPen('k')
        self.pw2.setRange(xRange=(0, 500), yRange=(0, 1.1))
        self.lr2 = pg.LinearRegionItem()
        self.lr2.setBounds((0, 500))
        self.il1 = pg.InfiniteLine()
        self.il1.setPen('g', width=2)
        self.il2 = pg.InfiniteLine()
        self.il2.setPen('k')
        # Insertion des fonction sur le layout
        mainLayout.addWidget(self.nomFichier, 0, 0, 1, 4)
        mainLayout.addWidget(self.pw1, 1, 0, 6, 4)
        mainLayout.addWidget(btn5, 1, 4)
        mainLayout.addWidget(btn6, 2, 4)
        mainLayout.addWidget(btn7, 3, 4)
        mainLayout.addWidget(btn8, 4, 4)
        mainLayout.addWidget(self.pw2, 7, 0, 6, 3)
        mainLayout.addWidget(self.lbl1, 8, 3, 1, 2)
        mainLayout.addWidget(self.lbl2, 7, 3, 1, 2)
        mainLayout.addWidget(self.lbl6, 9, 3)
        mainLayout.addWidget(self.cb2, 9, 4)
        mainLayout.addWidget(lbl3, 10, 3, 1, 1)
        mainLayout.addWidget(self.cb1, 10, 4, 1, 1)
        mainLayout.addWidget(lbl4, 11, 3)
        mainLayout.addWidget(self.sb1, 12, 3)
        mainLayout.addWidget(lbl5, 11, 4)
        mainLayout.addWidget(self.sb2, 12, 4)

        mainLayout.addWidget(btn1, 13, 0)
        mainLayout.addWidget(btn2, 13, 1)
        mainLayout.addWidget(btn3, 13, 2)
        mainLayout.addWidget(btn4, 13, 3)
        mainLayout.addWidget(btn9, 13, 4)
        # Creation des interactions des boutons
        btn1.clicked.connect(self.OpenFile)
        btn2.clicked.connect(self.closeEvent)
        btn3.clicked.connect(self.PerformFilter)
        btn4.clicked.connect(self.ClearFilteredPlot)
        btn5.clicked.connect(lambda: self.ShowLinearRegion(self.pw1, self.lr1))
        btn6.clicked.connect(lambda: self.HideLinearRegion(self.pw1, self.lr1))
        btn7.clicked.connect(self.SelectData)
        btn8.clicked.connect(self.ClearSelectData)
        btn9.clicked.connect(self.UseFilteredData)
        # Autres interactions
        self.sb1.valueChanged.connect(lambda: self.UpdateLinearRegion(self.lr2,
                                                                      self.sb1.value(),
                                                                      self.sb2.value()))
        self.sb2.valueChanged.connect(lambda: self.UpdateLinearRegion(self.lr2,
                                                                      self.sb1.value(),
                                                                      self.sb2.value()))
        self.lr2.sigRegionChanged.connect(lambda: self.UpdateSpinBox(self.sb1,
                                                                     self.lr2.getRegion()[0]))
        self.lr2.sigRegionChanged.connect(lambda: self.UpdateSpinBox(self.sb2,
                                                                     self.lr2.getRegion()[1]))

    def OpenFile(self):
        """
        Permet de choisir un fichier, de l'ouvrir
        """
        global data
        self.HideLinearRegion(self.pw1, self.lr1)
        """Permet de choisir le fichier à ouvrir"""
        self.fd.setDirectory('W:/R-D/18R047_OOSSI/Création_édition_logiciels/Python/Ressources/')
        self.fileName = self.fd.getOpenFileName(filter='*.txt')[0]
        self.nomFichier.setText(self.fileName.split("/").pop())
        self.dataInit, self.freq = ImporterDonnees(self.fileName)
        self.data = np.copy(self.dataInit)
        self.unfilteredPlot.setData(self.data, name="Données brutes")
        self.lr1.setBounds((self.data[:, 0].min(), self.data[:, 0].max()))
        self.lr1.setRegion(np.array([0.25, 0.75])*self.data[:, 0].max())
        self.filteredPlot.clear()
        self.PerformFFT()

    def CloseWindow(self):
        self.close()
        QtWidgets.QApplication.quit()

    def closeEvent(self, event):
        self.CloseWindow()

    def PerformFFT(self):
        self.pw2.clear()
        self.pw2.addItem(self.lr2)
        freq_FFT, spectre_FFT = Calculer_FFT(self.data[:, 1], self.freq)
        self.pw2.plot(freq_FFT, spectre_FFT, pen='r')
        freq_res = freq_FFT[spectre_FFT.argmax()]
        self.lbl2.setText("Frequence de résonance = {} Hz".format(round(freq_res, 2)))
        self.lbl1.setText("Fréquence d'échantillonage = {} Hz".format(int(self.freq)))
        self.lr2.setRegion((self.sb1.value(), self.sb2.value()))
        self.sb2.setMaximum(freq_FFT.max())
        self.sb1.setMaximum(freq_FFT.max())

    def PerformFilter(self):
        if self.data is not None:
            filtered_data = Filtrer_Data(self.data[:, 1],
                                         self.freq,
                                         lowcut=self.sb1.value(),
                                         highcut=self.sb2.value(),
                                         order=int(self.cb1.currentText()),
                                         btype = self.cb2.currentText())
            self.filtered_data = np.column_stack((self.data[:, 0],
                                                  filtered_data))
            self.filteredPlot.setData(self.filtered_data,
                                      name="Données filtrées")
            self.PerformFFT()
        else:
            print("Il n'y a pas encore de donées à filtrer")

    def UpdateSpinBox(self, spinBox, value):
        spinBox.setValue(value)

    def UpdateLinearRegion(self, linearRegion, valueMin, valueMax):
        linearRegion.setRegion((valueMin, valueMax))

    def ClearFilteredPlot(self):
        self.filteredPlot.clear()

    def ShowLinearRegion(self, plotWidget, linearRegion):
        if self.data is not None:
            plotWidget.addItem(linearRegion)
        else:
            print("Choisir d'abord un fichier à traiter")

    def HideLinearRegion(self, plotWidget, linearRegion):
        plotWidget.removeItem(linearRegion)

    def SelectData(self):
        if self.data is not None:
            self.filteredPlot.clear()
            index = RechercheIndex(self.lr1.getRegion(), self.data[:, 0])
            self.data = self.data[index[0]:index[1], :]
            self.unfilteredPlot.setData(self.data)
            self.lr1.setBounds((self.data[:, 0].min(), self.data[:, 0].max()))
            self.PerformFFT()
        else:
            print("Choisir d'abord un fichier à traiter")

    def ClearSelectData(self):
        self.filteredPlot.clear()
        self.data = np.copy(self.dataInit)
        self.lr1.setBounds((self.data[:, 0].min(), self.data[:, 0].max()))
        self.unfilteredPlot.setData(self.data)
        self.PerformFFT()

    def UseFilteredData(self):
        self.data = self.filtered_data
        self.filteredPlot.clear()
        self.unfilteredPlot.setData(self.data)
        self.PerformFFT()

# =============================================================================
# Fonction intermédiaires
# =============================================================================
def ImporterDonnees(fileName):
    """Ouvre le fichier et récupère les données de déplacement.
    Ouvre ensuite le fichier paramètres correspondant et récupère la fréquence
    de l'essai.\n
    Parameters
    ----------
    fileName : string
            Nom du fichier d'entrée
    Return
    ------
    out : ndarray
        Tableau [temps, déplacement]
    frequence : int
        Frequence echantillonage"""
#    global frequence, tps, dep
    dep = np.loadtxt(fileName)
    fileParams = fileName.split(".")[:-1][0]
    with open("{}_param.txt".format(fileParams), "r") as fichier:
        content = fichier.read()
    tab = content.split("\n")
    frequence = int(tab[1].split(": ")[-1])
    tps = np.linspace(0, len(dep)-1, len(dep))*1/frequence
    out = np.column_stack((tps, dep))
    return out, frequence


def Filtrer_Data(data, fs, lowcut=0, highcut=0, order=2, btype='lowpass'):
    """
    Applique un filtre de butterworth sur les données.\n
    Parameters
    ------------
    data : 1dArray
        Tableau de données à filtrer
    fs : float
        Fréquence d'échantillonage (Hz)'
    lowcut : float
        Fréquence de coupure basse
    highcut : float
        Fréquence de coupure haute (Hz)
    order : int
        Ordre du filtre
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        Type du filtre. 'lowpass' est l'option par défaut'
    Return
    -------
    filtered_data : 1dArray
        Tabelau des données filtrées
    """
    sos = Calculer_Parametres_Butter(fs,
                                     lowcut=lowcut,
                                     highcut=highcut,
                                     order=order,
                                     btype=btype)
    filtered_data = sosfilt(sos, data)
    return filtered_data


def Calculer_Parametres_Butter(fs, lowcut=1e-9, highcut=1e9, order=2,
                               btype="lowpass"):
    """
    Calcule les paramètres d'un filtre Butterworth.\n
    Parameters
    ----------
    lowcut : float
        Fréquence de coupure basse
    highcut : float
        Fréquence de coupure haute (Hz)
    fs : float
        Fréquence d'échantillonage (Hz)'
    order : int, Optional
        Ordre du filtre
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        Type du filtre. 'lowpass' est l'option par défaut'
    Return
    -------
    sos : ndarray
        Parametre du filtre IR.
    """
    nyq = fs/2
    low = lowcut / nyq
    high = highcut / nyq
    if btype == "lowpass":
        Wn = high
    elif btype == "highpass":
        Wn = low
    else:
        Wn = [low, high]
    sos = butter(order, Wn, btype=btype, output='sos')
    return sos


def Calculer_FFT(data, frequence):
    """Calcul de la FFT.\n
    Parameters
    ----------
    data : ndarray
        tableau 1D contenant les données pour le calcul de la FFT
    frequence : int
        Fréquence d'échantillonage
    Return
    ------
    spectre : ndarray
        Spectre de fréquence issue de la FFT
    freq_FFT : ndarray
        Tableau de fréquences associées au spectre FFT
    """
    sample = len(data)//2
    spectre = abs(np.fft.fft(data)/np.fft.fft(data).max()).real[:sample]
    freq_FFT = np.fft.fftfreq(len(data), 1/frequence)[:sample]
    return freq_FFT, spectre


def RechercheIndex(tabValeurs, tabRecherche):
    """
    Recherche les index du tableau les plus proches de valeurs données.\n
    Parameters
    -----------
    tabValeurs : ndarray
        Tableau 1D contenant les valeurs dont les indexs doivent être déterminées
    tabRecherche : ndarray
        Tableau 1D du tableau dans lequel doit être cherchés les indexs
    Return
    -------
    tabIndex: ndarray
        Tableau 1D contenant les indexs
    """
    tabIndex = []
    for valeur in tabValeurs:
        tabIndex.append(np.abs(tabRecherche - valeur).argmin())
    return tabIndex


# =============================================================================
# Execution programme
# =============================================================================
if __name__ == '__main__':
    screen = MyApp()
    screen.show()
    app.exec_()


app.quit()
#if __name__ == "__main__":
#    myApp = MyApp()
#    myApp.show()
#    app.exec_()
