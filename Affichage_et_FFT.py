# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:59:26 2019

@author: t.fourcade

Logiciel pour l'affichage, le calcul de FFT et le filtrage
"""

import numpy as np
from PyQt5 import QtWidgets, QtGui
import pyqtgraph as pg
from scipy.signal import butter, sosfilt, cheby1, cheby2, ellip, bessel
from pathlib import Path


if not QtWidgets.QApplication.instance():
    app = QtWidgets.QApplication([])
else:
    app = QtWidgets.QApplication.instance()


pg.setConfigOption("background", "w")


class MyApp(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super(MyApp, self).__init__()
        # QtWidgets.QMainWindow.__init__(self)
        """Initialisation de la fenêtre principale"""
        self.resize(1200, 800)
        self.setWindowTitle("Traitement des données")
        # Initilisation des données
        self.data = None
        # Ajout d'un widget central
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        # Création des layouts
        main_layout = QtWidgets.QGridLayout()
        central_widget.setLayout(main_layout)
        filter_layout = QtWidgets.QGridLayout()
        select_layout = QtWidgets.QGridLayout()
        # Creation des boutons et labels
        self.nom_fichier = QtWidgets.QLabel()
        btn_ouvrir = QtWidgets.QPushButton("Ouvrir")
        btn_fermer = QtWidgets.QPushButton("Fermer")
        btn_filtrer = QtWidgets.QPushButton("Filtrer")
        btn_supprimer_filtre = QtWidgets.QPushButton("Supprimer\nfiltre")
        btn_afficher_filtre = QtWidgets.QPushButton("Filtre de sélection")
        btn_cacher_filtre_selection = QtWidgets.QPushButton("Cacher filtre de sélection")
        btn_selectionner_donner = QtWidgets.QPushButton("Sélectionner données")
        btn_annuler_selection = QtWidgets.QPushButton("Annuler sélection")
        btn_utiliser_donnees_filtrees = QtWidgets.QPushButton("Utiliser données filtrées")
        self.fd = QtWidgets.QFileDialog(filter="*.txt")
        self.lbl1 = QtWidgets.QLabel("Fréquence d'échantillonage = {} Hz".format(0))
        self.lbl2 = QtWidgets.QLabel("Frequence de résonance = {} Hz".format(0))
        lbl3 = QtWidgets.QLabel("Ordre du filtre")
        self.cb1 = QtWidgets.QComboBox()
        self.cb1.addItems(["2", "3", "4", "5", "6", "7", "8", "9", "10"])
        self.cb1.setCurrentText("8")
        lbl4 = QtWidgets.QLabel("Frequence de coupure basse (Hz)")
        self.sb1 = pg.SpinBox()
        self.sb1.setValue(40)
        self.sb1.setMinimum(0)
        self.sb1.setDecimals(4)
        self.sb1.setSingleStep(0.1)
        lbl5 = QtWidgets.QLabel("Frequence de coupure haute (Hz)")
        self.sb2 = pg.SpinBox()
        self.sb2.setValue(400)
        self.sb2.setMinimum(1)
        self.sb2.setSingleStep(0.1)
        self.sb2.setOpts(minStep=0.1)
        self.sb2.setDecimals(4)
        self.lbl6 = QtWidgets.QLabel("Type de filtre")
        self.cb2 = QtWidgets.QComboBox()
        self.cb2.addItems(["bandpass", "lowpass", "highpass", "bandstop"])
        lbl7 = QtWidgets.QLabel("Classe du filtre")
        self.cb3 = QtWidgets.QComboBox()
        self.cb3.addItems(
            [
                "Butterworth",
                "Chebyshev ordre 1",
                "Chebyshev ordre 2",
                "Elliptic",
                "Bessel",
            ]
        )
        self.lbl8 = QtWidgets.QLabel("rs")
        self.le1 = QtWidgets.QLineEdit()
        self.le1.setDisabled(True)
        validation_donnees = QtGui.QDoubleValidator(0, 100, 2)
        self.le1.setValidator(validation_donnees)
        self.le1.setText("1")
        self.lbl9 = QtWidgets.QLabel("rp")
        self.le2 = QtWidgets.QLineEdit()
        self.le2.setValidator(validation_donnees)
        self.le2.setDisabled(True)
        self.le2.setText("2")
        self.lbl10 = QtWidgets.QLabel("Normalisation")
        self.cb4 = QtWidgets.QComboBox()
        self.cb4.addItems(["Phase", "Delay", "Mag"])
        self.cb4.setDisabled(True)
        # Creation du graph d'affichage de la courbe temps-déplacement
        self.pw1 = pg.PlotWidget(title="Temps-Déplacement")
        self.pw1.getPlotItem().showGrid(x=True, y=True)
        self.pw1.getPlotItem().getAxis("bottom").setPen("k")
        self.pw1.getPlotItem().getAxis("left").setPen("k")
        self.pw1.getPlotItem().setLabel("bottom", "Temps", "s")
        self.pw1.getPlotItem().setLabel("left", "Déplacemen", "nm")
        self.unfiltered_plot = self.pw1.getPlotItem().plot()
        self.filtered_plot = self.pw1.getPlotItem().plot()
        self.filtered_plot.setPen("g")
        self.unfiltered_plot.setPen("b")
        self.lr1 = pg.LinearRegionItem()
        # Creation du graph d'affichage de la courbe de FFT
        self.pw2 = pg.PlotWidget(title="FFT")
        self.pw2.getPlotItem().showGrid(x=True, y=True)
        self.pw2.getPlotItem().getAxis("bottom").setPen("k")
        self.pw2.getPlotItem().getAxis("left").setPen("k")
        self.pw2.setXRange(0, 2000)
        self.pw2.setYRange(0, 1.1)
        self.pw2.setLimits(xMin=0, xMax=2000)
        self.lr2 = pg.LinearRegionItem()
        self.lr2.setBounds((0, 2000))
        self.il1 = pg.InfiniteLine()
        self.il1.setPen("g", width=2)
        self.il2 = pg.InfiniteLine()
        self.il2.setPen("k")
        # Insertion des boutons sur  main_layout
        main_layout.addWidget(self.nom_fichier, 0, 0, 1, 4)
        main_layout.addWidget(self.pw1, 1, 0, 1, 4)
        main_layout.addWidget(self.pw2, 2, 0, 1, 3)
        main_layout.addWidget(btn_ouvrir, 3, 0)
        main_layout.addWidget(btn_fermer, 3, 1)
        main_layout.addWidget(btn_filtrer, 3, 2)
        main_layout.addWidget(btn_supprimer_filtre, 3, 3)
        main_layout.addWidget(btn_utiliser_donnees_filtrees, 3, 4)
        main_layout.addLayout(select_layout, 1, 4)
        main_layout.addLayout(filter_layout, 2, 3, 1, 2)
        # Ajout des boutons sur select_layout
        select_layout.addWidget(btn_afficher_filtre, 1, 1)
        select_layout.addWidget(btn_cacher_filtre_selection, 2, 1)
        select_layout.addWidget(btn_selectionner_donner, 3, 1)
        select_layout.addWidget(btn_annuler_selection, 4, 1)
        # Ajout des boutons sur filter_layout
        filter_layout.addWidget(self.lbl1, 1, 1, 1, 2)
        filter_layout.addWidget(self.lbl2, 2, 1, 1, 2)
        filter_layout.addWidget(self.lbl6, 3, 1)
        filter_layout.addWidget(self.cb2, 3, 2)
        filter_layout.addWidget(lbl7, 4, 1)
        filter_layout.addWidget(self.cb3, 4, 2)
        filter_layout.addWidget(lbl3, 5, 1)
        filter_layout.addWidget(self.cb1, 5, 2)
        filter_layout.addWidget(lbl4, 6, 1)
        filter_layout.addWidget(self.sb1, 6, 2)
        filter_layout.addWidget(lbl5, 7, 1)
        filter_layout.addWidget(self.sb2, 7, 2)
        filter_layout.addWidget(self.lbl8, 8, 1)
        filter_layout.addWidget(self.le1, 8, 2)
        filter_layout.addWidget(self.lbl9, 9, 1)
        filter_layout.addWidget(self.le2, 9, 2)
        filter_layout.addWidget(self.lbl10, 10, 1)
        filter_layout.addWidget(self.cb4, 10, 2)
        # Creation des interactions des boutons
        btn_ouvrir.clicked.connect(self.open_file)
        btn_fermer.clicked.connect(self.closeEvent)
        btn_filtrer.clicked.connect(self.perform_filter)
        btn_supprimer_filtre.clicked.connect(self.clear_filtered_plot)
        btn_afficher_filtre.clicked.connect(lambda: self.show_linear_region(self.pw1, self.lr1))
        btn_cacher_filtre_selection.clicked.connect(
            lambda: self.hide_linear_region(self.pw1, self.lr1)
        )
        btn_selectionner_donner.clicked.connect(self.select_data)
        btn_annuler_selection.clicked.connect(self.clear_selected_data)
        btn_utiliser_donnees_filtrees.clicked.connect(self.use_filtered_data)
        # Autres interactions
        self.sb1.sigValueChanged.connect(
            lambda: self.update_linear_region(self.lr2, self.sb1.value(), "min")
        )
        self.sb2.sigValueChanged.connect(
            lambda: self.update_linear_region(self.lr2, self.sb2.value(), "max")
        )
        self.lr2.sigRegionChanged.connect(self.update_spin_boxes)
        self.cb3.currentIndexChanged.connect(self.define_class_filter_options)
        self.cb2.currentIndexChanged.connect(self.on_cb2_activated)

    def open_file(self) -> None:
        """
        Permet de choisir un fichier, de l'ouvrir de tracer le graphe et la FFT
        """
        self.hide_linear_region(self.pw1, self.lr1)
        """Permet de choisir le fichier à ouvrir"""
        self.fd.setDirectory("W:/R-D/18R047_OOSSI/Création_édition_logiciels/Python/Ressources/")
        self.fileName = self.fd.getOpenFileName(filter="*.txt *.csv")[0]
        self.nom_fichier.setText(self.fileName.split("/").pop())
        self.dataInit, self.freq = importer_donnees(self.fileName)
        self.data = np.copy(self.dataInit)
        self.unfiltered_plot.setData(self.data, name="Données brutes")
        self.lr1.setBounds((self.data[:, 0].min(), self.data[:, 0].max()))
        self.lr1.setRegion(np.array([0.25, 0.75]) * self.data[:, 0].max())
        self.define_class_filter_options()
        self.filtered_plot.clear()
        self.perform_FFT()

    def close_window(self) -> None:
        """
        Quitte proprement l'application
        """
        self.close()
        QtWidgets.QApplication.quit()

    def closeEvent(self, event) -> None:
        self.close_window()

    def perform_FFT(self) -> None:
        """
        Calcule la FFT et la trace
        """
        self.pw2.clear()
        self.pw2.addItem(self.lr2)
        freq_FFT, spectre_FFT = calculer_FFT(self.data[:, 1] - self.data[:, 1].mean(), self.freq)
        self.pw2.plot(freq_FFT, spectre_FFT, pen="r")
        freq_res = freq_FFT[spectre_FFT.argmax()]
        self.lbl2.setText(f"Frequence de résonance = {freq_res: 0.2f} Hz")
        self.lbl1.setText(f"Fréquence d'échantillonage = {int(self.freq)} Hz")
        self.lr2.setRegion((self.sb1.value(), self.sb2.value()))
        self.sb2.setMaximum(freq_FFT.max())
        self.sb1.setMaximum(freq_FFT.max())

    def perform_filter(self) -> None:
        """
        Applique le filtre choisi sur les données sélectionnées
        """
        if self.sb1.value() == 0:
            low_cut = 1e-3
        else:
            low_cut = self.sb1.value()
        high_cut = self.sb2.value
        if self.data is not None:
            if self.cb3.currentIndex() == 0:
                filtered_data = filtrer_data_butter(
                    self.data[:, 1],
                    self.freq,
                    lowcut=low_cut,
                    highcut=self.sb2.value(),
                    order=int(self.cb1.currentText()),
                    band_type=self.cb2.currentText(),
                )
            elif self.cb3.currentIndex() == 1:
                filtered_data = filtrer_data_cheby1(
                    self.data[:, 1],
                    self.freq,
                    float(self.le1.text()),
                    lowcut=low_cut,
                    highcut=high_cut,
                    order=int(self.cb1.currentText()),
                    band_type=self.cb2.currentText(),
                )
            elif self.cb3.currentIndex() == 2:
                filtered_data = filtrer_data_cheby2(
                    self.data[:, 1],
                    self.freq,
                    float(self.le2.text()),
                    lowcut=low_cut,
                    highcut=high_cut,
                    order=int(self.cb1.currentText()),
                    band_type=self.cb2.currentText(),
                )
            elif self.cb3.currentIndex() == 3:
                filtered_data = filtrer_data_ellip(
                    self.data[:, 1],
                    self.freq,
                    float(self.le1.text()),
                    float(self.le2.text()),
                    lowcut=low_cut,
                    highcut=high_cut,
                    order=int(self.cb1.currentText()),
                    band_type=self.cb2.currentText(),
                )
            else:
                filtered_data = filtrer_data_bessel(
                    self.data[:, 1],
                    self.freq,
                    lowcut=low_cut,
                    highcut=high_cut,
                    order=int(self.cb1.currentText()),
                    band_type=self.cb2.currentText(),
                    norm=self.cb4.currentText(),
                )
            self.filtered_data = np.column_stack((self.data[:, 0], filtered_data))
            self.filtered_plot.setData(self.filtered_data, name="Données filtrées")
            self.perform_FFT()
        else:
            print("Il n'y a pas encore de données à filtrer")

    def update_spin_boxes(self) -> None:
        """
        Met à jour les valeurs des spinbox représentant les valeurs
        hautes et basses des limite de filtrage.
        """
        self.sb1.setValue(self.lr2.getRegion()[0])
        self.sb2.setValue(self.lr2.getRegion()[1])

    def update_linear_region(
        self, linear_region: pg.LinearRegionItem, value: float, type: str
    ) -> None:
        """
        Met à jour la limite de la linear region en fonction de
        la valeur de la spinbox correspondante.
        Parameters:
        ------------
        linear_region : linear Region\n
            \tzone de sélection à mettre à jour\n
        value : double\n
            \tnouvelle valeur pour la zone de sélection\n
        type : (min, max)
            \tLimite haute ou basse à modifier.\n
        """
        if type == "min":
            linear_region.childItems()[0].setValue(value)
        elif type == "max":
            linear_region.childItems()[1].setValue(value)

    def clear_filtered_plot(self):
        """
        Supprime l'affichage des données filtrées.
        """
        self.filtered_plot.clear()

    def show_linear_region(
        self, plot_widget: pg.PlotWidget, linearRegion: pg.LinearRegionItem
    ) -> None:
        """
        Affiche la région de sélection dans le plot sélectionné
        Parameters:

        Parameters
        ----------
        plot_widget : PlotWidget
            PlotWidget dans lequel la Linear Region doit être affichée

        linearRegion : LinearRegion
            LinearRegion à afficher dans plot_widget
        """
        if self.data is not None:
            plot_widget.addItem(linearRegion)
        else:
            print("Choisir d'abord un fichier à traiter")

    def hide_linear_region(
        self, plot_widget: pg.PlotWidget, linearRegion: pg.LinearRegionItem
    ) -> None:
        """
        Cache la région de sélection dans le plot sélectionné
        Parameters:
        -----------
        plot_widget
        linear_region
        """
        plot_widget.removeItem(linearRegion)

    def select_data(self):
        """
        Sélectionne les données dans le filtre de sélection.
        """
        if self.data is not None:
            self.filtered_plot.clear()
            index = recherche_index(self.lr1.getRegion(), self.data[:, 0])
            self.data = self.data[index[0] : index[1], :]
            self.unfiltered_plot.setData(self.data)
            self.lr1.setBounds((self.data[:, 0].min(), self.data[:, 0].max()))
            self.perform_FFT()
        else:
            print("Choisir d'abord un fichier à traiter")

    def clear_selected_data(self) -> None:
        """
        Réinitialise les données (filtre et sélection)
        """
        self.filtered_plot.clear()
        self.data = np.copy(self.dataInit)
        self.lr1.setBounds((self.data[:, 0].min(), self.data[:, 0].max()))
        self.unfiltered_plot.setData(self.data)
        self.perform_FFT()

    def use_filtered_data(self) -> None:
        """
        Permet d'utiliser les données filtrées pour réaliser un traitement
        """
        self.data = self.filtered_data
        self.filtered_plot.clear()
        self.unfiltered_plot.setData(self.data)
        self.perform_FFT()

    def define_class_filter_options(self) -> None:
        """
        Initialise les paramètre pour les différents type de filtre
        """
        if self.cb3.currentText() == "Butterworth":
            self.le1.setDisabled(True)
            self.le2.setDisabled(True)
            self.cb4.setDisabled(True)
        elif self.cb3.currentText() == "Chebyshev ordre 1":
            self.le1.setEnabled(True)
            self.le2.setDisabled(True)
            self.cb4.setDisabled(True)
        elif self.cb3.currentText() == "Chebyshev ordre 2":
            self.le1.setDisabled(True)
            self.le2.setEnabled(True)
            self.cb4.setDisabled(True)
        elif self.cb3.currentText() == "Elliptic":
            self.le1.setEnabled(True)
            self.le2.setEnabled(True)
            self.cb4.setDisabled(True)
        else:
            self.le1.setDisabled(True)
            self.le2.setDisabled(True)
            self.cb4.setEnabled(True)

    def on_cb2_activated(self) -> None:
        """
        Initialise les données du filtre en fonction du type choisi
        (bandpass, stopband, lowpass, highpass)
        """
        if self.cb2.currentText() == "lowpass":
            self.sb1.setDisabled(True)
            self.sb2.setEnabled(True)
            self.sb1.setValue(0)
            self.lr2.setMovable(False)
            self.lr2.childItems()[1].setMovable(True)
        elif self.cb2.currentText() == "highpass":
            self.sb1.setEnabled(True)
            self.sb2.setDisabled(True)
            self.sb2.setValue(500)
            self.lr2.setMovable(False)
            self.lr2.childItems()[0].setMovable(True)
        else:
            self.sb1.setEnabled(True)
            self.sb2.setEnabled(True)
            self.lr2.setMovable(True)
            # for line in self.lr2.childItems():
            #     line.setMovable(True)


# =============================================================================
# Fonction intermédiaires
# =============================================================================
def importer_donnees(fileName: str) -> tuple[np.ndarray, int]:
    """Ouvre le fichier et récupère les données de temps déplacement.
    Si un fichier paramètre correspondant existe, l'ouvre et et récupère la fréquence et reconstruit
    les données de temps.
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
    file_name = Path(fileName).stem
    if Path(fileName).with_name(file_name + "_param.txt").exists():
        dep = np.loadtxt(fileName)
        file_params = fileName.split(".")[:-1][0]
        with open("{}_param.txt".format(file_params), "r") as fichier:
            content = fichier.read()
        tab = content.split("\n")
        frequence = int(tab[1].split(": ")[-1])
        tps = np.linspace(0, len(dep) - 1, len(dep)) * 1 / frequence
        out = np.column_stack((tps, dep))
    else:
        first_line = Path(fileName).read_text().splitlines()[0]
        if first_line.count(";") > 0:
            delimiter = ";"
        elif first_line.count(",") > 0:
            delimiter = ","
        elif first_line.count("\t") > 0:
            delimiter = "\t"
        else:
            print(
                "Les colonnes ne sont séparées par aucun delimiteur connu. Remplacer le delimiteur"
            )
            raise IndexError

        if all([val.isnumeric() for val in first_line.split(delimiter)]):
            out = np.loadtxt(fileName, delimiter=delimiter)
        else:
            out = np.loadtxt(fileName, skiprows=1, delimiter=delimiter, usecols=[0, 1])
        frequence = 1 / np.diff(out[:, 0]).mean()

    return out, frequence


def filtrer_data_butter(
    data: np.ndarray,
    fs: float,
    lowcut: float = 0,
    highcut: float = 0,
    order: int = 2,
    band_type: str = "lowpass",
) -> np.ndarray:
    """
    Applique un filtre de butterworth sur les données.\n
    Parameters
    ------------
    data : 1dArray
        Tableau de données à filtrer
    fs : float
        Fréquence d'échantillonage (Hz)
    lowcut : float
        Fréquence de coupure basse
    highcut : float
        Fréquence de coupure haute (Hz)
    order : int
        Ordre du filtre
    band_type : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        Type du filtre. 'lowpass' est l'option par défaut'
    Return
    -------
    filtered_data : 1dArray
        Tabelau des données filtrées
    """
    Wn = normalisation_frequence(fs, lowcut, highcut, band_type)
    sos = butter(order, Wn, btype=band_type, output="sos")
    filtered_data: np.ndarray = sosfilt(sos, data)
    return filtered_data


def filtrer_data_cheby1(
    data: np.ndarray,
    fs: float,
    rp: float,
    lowcut: float = 1e-9,
    highcut: float = 1e9,
    order: int = 2,
    band_type: str = "lowpass",
) -> np.ndarray:
    """
    Calcule les paramètres d'un filtre Chebyshev de type I\n
    Parameters
    ------------
    data : 1dArray
        Tableau de données à filtrer
    fs : float
        Fréquence d'échantillonage (Hz)
    rs : float
        Ondulation maximale autorisée dans la bande passante (dB)
    lowcut : float
        Fréquence de coupure basse
    highcut : float
        Fréquence de coupure haute (Hz)
    order : int
        Ordre du filtre
    band_type : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        Type du filtre. 'lowpass' est l'option par défaut'
    Return
    -------
    filtered_data : 1dArray
        Tabelau des données filtrées
    """
    Wn = normalisation_frequence(fs, lowcut, highcut, band_type)
    sos = cheby1(order, rp, Wn, btype=band_type, output="sos")
    filtered_data: np.ndarray = sosfilt(sos, data)
    return filtered_data


def filtrer_data_cheby2(
    data: np.ndarray,
    fs: float,
    rs: float,
    lowcut: float = 1e-9,
    highcut: float = 1e9,
    order: int = 2,
    band_type: str = "lowpass",
) -> np.ndarray:
    """
    Calcule les paramètres d'un filtre Chebyshev de type II\n
    Parameters
    ------------
    data : 1dArray
        Tableau de données à filtrer
    fs : float
        Fréquence d'échantillonage (Hz)
    rp : float
        Ondulation maximale requise dans la bande coupée (dB)
    lowcut : float
        Fréquence de coupure basse
    highcut : float
        Fréquence de coupure haute (Hz)
    order : int
        Ordre du filtre
    band_type : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        Type du filtre. 'lowpass' est l'option par défaut'
    Return
    -------
    filtered_data : 1dArray
        Tabelau des données filtrées
    """
    Wn = normalisation_frequence(fs, lowcut, highcut, band_type)
    sos = cheby2(order, rs, Wn, btype=band_type, output="sos")
    filtered_data: np.ndarray = sosfilt(sos, data)
    return filtered_data


def filtrer_data_ellip(
    data: np.ndarray,
    fs: float,
    rs: float,
    rp: float,
    lowcut: float = 1e-9,
    highcut: float = 1e9,
    order: float = 2,
    band_type: str = "lowpass",
) -> np.ndarray:
    """
    Calcule les paramètres d'un filtre Elliptique\n
    Parameters
    ------------
    data : 1dArray
        Tableau de données à filtrer
    fs : float
        Fréquence d'échantillonage (Hz)
    rs : float
        Ondulation maximale autorisée dans la bande passante (dB)
    rp : float
        Ondulation maximale requise dans la bande coupée (dB)
    lowcut : float
        Fréquence de coupure basse
    highcut : float
        Fréquence de coupure haute (Hz)
    order : int
        Ordre du filtre
    band_type : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        Type du filtre. 'lowpass' est l'option par défaut.
    Return
    -------
    filtered_data : 1dArray
        Tabelau des données filtrées
    """
    Wn = normalisation_frequence(fs, lowcut, highcut, band_type)
    sos = ellip(order, rs, rp, Wn, btype=band_type, output="sos")
    filtered_data = sosfilt(sos, data)
    return filtered_data


def filtrer_data_bessel(
    data: np.ndarray,
    fs: float,
    lowcut: float = 0,
    highcut: float = 0,
    order: int = 2,
    band_type: str = "lowpass",
    norm: str = "phase",
) -> np.ndarray:
    """
    Applique un filtre de Bessel sur les données.\n
    Parameters
    ------------
    data : 1dArray
        Tableau de données à filtrer

    fs : float
        Fréquence d'échantillonage (Hz)

    lowcut : float
        Fréquence de coupure basse

    highcut : float
        Fréquence de coupure haute (Hz)

    order : int
        Ordre du filtre

    band_type : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        Type du filtre. 'lowpass' est l'option par défaut'

    norm : {'phase', 'delay', 'mag'}
    Critical frequency normalization:\n
        \tphase : The filter is normalized such that the phase response
        \treaches its midpoint at angular (e.g. rad/s) frequency Wn. This
        \thappens for both low-pass and high-pass filters, so this is the
        \t“phase-matched” case.

        \tdelay : The filter is normalized such that the group delay in the
        \tpassbandis 1/Wn (e.g. seconds). This is the “natural” type obtained
        \tby solving Bessel polynomials.

        \tmag : The filter is normalized such that the gain magnitude is -3 dB
        \tat angular frequency Wn.

    Return
    -------
    filtered_data : 1dArray
        Tabelau des données filtrées
    """
    Wn = normalisation_frequence(fs, lowcut, highcut, band_type)
    sos = bessel(order, Wn, btype=band_type, norm=norm, output="sos")
    filtered_data: np.ndarray = sosfilt(sos, data)
    return filtered_data


def normalisation_frequence(
    fs: float, lowcut: float, highcut: float, band_type: str
) -> float | list[float]:
    """
    Normalise les fréquences pour l'utilisation d'un filtre digital
    """
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    if band_type == "lowpass":
        Wn = high
    elif band_type == "highpass":
        Wn = low
    else:
        Wn = [low, high]
    return Wn


def calculer_FFT(data: np.ndarray, frequence: float) -> tuple[np.ndarray, np.ndarray]:
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
    sample = len(data) // 2
    spectre = abs(np.fft.fft(data) / np.fft.fft(data).max()).real[:sample]
    freq_FFT = np.fft.fftfreq(len(data), 1 / frequence)[:sample]
    return freq_FFT, spectre


def recherche_index(tab_valeurs: np.ndarray, tab_recherche: np.ndarray) -> list:
    """
    Recherche les index du tableau les plus proches de valeurs données.\n
    Parameters
    -----------
    tab_valeurs : ndarray
        Tableau 1D contenant les valeurs dont les indexs doivent être
        déterminées
    tab_recherche : ndarray
        Tableau 1D du tableau dans lequel doit être cherchés les indexs
    Return
    -------
    tabIndex: list
        Tableau 1D contenant les indexs
    """
    tabIndex = []
    for valeur in tab_valeurs:
        tabIndex.append(np.abs(tab_recherche - valeur).argmin())
    return tabIndex


# =============================================================================
# Execution programme
# =============================================================================
if __name__ == "__main__":
    screen = MyApp()
    screen.show()
    app.exec_()


app.quit()
# if __name__ == "__main__":
#    myApp = MyApp()
#    myApp.show()
#    app.exec_()
