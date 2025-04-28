# LabToolbox

**LabToolbox** è una libreria Python contenente una serie di funzioni utili per l'analisi di dati di laboratorio. Fornisce funzioni intuitive e ottimizzate per il fitting, la propagazione delle incertezze, la gestione dei dati e la visualizzazione grafica, rendendo più rapido e rigoroso il trattamento dei dati di laboratorio. Pensata per studenti, ricercatori e chiunque lavori con dati sperimentali, combina semplicità d'uso con rigore metodologico.

Il file `example.ipynb`, disponibile sulla pagina [GitHub](https://github.com/giusesorrentino/LabToolbox) della libreria, contiene esempi di utilizzo delle principali funzioni di `LabToolbox`.

## Disclaimer

Le funzioni `my_cov`, `my_var`, `my_mean`, `my_line`, `my_lin_fit` e `y_estrapolato` presenti nei moduli `LabToolbox.basics` e `LabToolbox.fit` fanno parte della libreria `my_lib_santanastasio`, sviluppata da F. Santanastasio (professore del corso di *Laboratorio di Meccanica* presso l'Università di Roma "La Sapienza"). Queste funzioni sono disponibili sul sito [https://baltig.infn.it/LabMeccanica/PythonJupyter](https://baltig.infn.it/LabMeccanica/PythonJupyter).

Inoltre, questo pacchetto fa uso della libreria `uncertainty_class` disponibile su [GitHub](https://github.com/yiorgoskost/Uncertainty-Propagation/tree/master). Fornisce funzionalità per la propagazione delle incertezze nei calcoli. Non è necessaria la sua installazione manuale, in quanto è presente come modulo in `LabToolbox`.

Le funzioni `lin_fit` e `model_fit` includono un'opzione per visualizzare i residui del fit. Il codice responsabile di questa funzionalità proviene dalla libreria [**VoigtFit**](https://github.com/jkrogager/VoigtFit).

## Installazione

Puoi installare **LabToolbox** facilmente utilizzando `pip`:

```bash
pip install LabToolbox