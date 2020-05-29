import sys
sys.path.append('/Users/Marco/miniconda3/envs/dl-proj3/lib/python3.7/site-packages')

import os
import re
import time
import nltk
# import string
import tensorlayer as tl
from utileria import *

import pickle


def GuardarPickle(targets, file):
    with open(file, 'wb') as f:
        pickle.dump(targets, f)

str_DirActual = os.getcwd()
str_DirImag = os.path.join(str_DirActual, 'imagenes')
str_DirFrases = os.path.join(str_DirActual, 'descripciones')
str_RutaVocabulario = str_DirActual + '/vocab.txt'

dict_Frases = {}
list_FrasesProcesadas = []
with tl.ops.suppress_stdout():
    files = tl.files.load_file_list(path=str_DirFrases, regx='^image_[0-9]+\.txt')
    for i, str_NombreArch in enumerate(files):
        str_RutaArch = os.path.join(str_DirFrases, str_NombreArch)
        key = int(re.findall('\d+', str_NombreArch)[0])
        file_Frases = open(str_RutaArch,'r')
        lista_Frases = []
        for str_Frase in file_Frases:
            str_Frase = PrepararFrase(str_Frase)
            lista_Frases.append(str_Frase)
            list_FrasesProcesadas.append(tl.nlp.process_sentence(str_Frase, start_word="<S>", end_word="</S>"))
        assert len(lista_Frases) == 5, "Cada imagen debe tener 5 descripciones"
        dict_Frases[key] = lista_Frases
print(" * %d x %d frases encontradas " % (len(dict_Frases), len(lista_Frases)))

if not os.path.isfile('vocab.txt'):
    _ = tl.nlp.create_vocab(list_FrasesProcesadas, word_counts_output_file=str_RutaVocabulario, min_word_count=1)
else:
    print("WARNING: vocab.txt already exists")
tensor_Vocabulario = tl.nlp.Vocabulary(str_RutaVocabulario, start_word="<S>", end_word="</S>", unk_word="<UNK>")

lista_Id_Frases = []
lista_Id_Frases = []
try: # python3
    tmp = dict_Frases.items()
except: # python3
    tmp = dict_Frases.iteritems()
for key, value in tmp:
    for v in value:
        lista_Id_Frases.append( [tensor_Vocabulario.word_to_id(word) for word in nltk.tokenize.word_tokenize(v)] + [tensor_Vocabulario.end_id])  # add END_ID

arr_Id_Frases = np.asarray(lista_Id_Frases)
print(" * tokenized %d captions" % len(arr_Id_Frases))

with tl.ops.suppress_stdout():
    list_NombresArchivos_Imagenes = sorted(tl.files.load_file_list(path=str_DirImag, regx='^image_[0-9]+\.jpg'))
print(" * %d imágenes encontradas, se inicia carga y escalamiento ..." % len(list_NombresArchivos_Imagenes))
s = time.time()

list_Imagenes = []
for str_NombreArchivo in list_NombresArchivos_Imagenes:
    img_raw = scipy.misc.imread( os.path.join(str_DirImag, str_NombreArchivo) )
    img = tl.prepro.imresize(img_raw, size=[64, 64])
    img = img.astype(np.float32)
    list_Imagenes.append(img)

print(" * la carga y escalamiento tomó %ss" % (time.time()-s))

nbr_Cant_Imag = len(dict_Frases)
nbr_Cant_Frases = len(arr_Id_Frases)
nbr_Cant_Frases_X_Imag = len(lista_Frases) # 10

print("nbr_Cant_Frases: %d nbr_Cant_Imag: %d nbr_Cant_Frases_X_Imag: %d" % (nbr_Cant_Frases, nbr_Cant_Imag, nbr_Cant_Frases_X_Imag))

arr_Id_Frases_train, arr_Id_Frases_test = arr_Id_Frases[: 8000*nbr_Cant_Frases_X_Imag], arr_Id_Frases[8000*nbr_Cant_Frases_X_Imag :]
arr_Imagenes_Train, arr_Imagenes_Test = list_Imagenes[:8000], list_Imagenes[8000:]

nbr_CantImag_Train = len(arr_Imagenes_Train)
nbr_CantImag_Test = len(arr_Imagenes_Test)
nbr_CantFrases_Train = len(arr_Id_Frases_train)
nbr_CantFrases_Test = len(arr_Id_Frases_test)
print("nbr_CantImag_Train:%d nbr_CantFrases_Train:%d" % (nbr_CantImag_Train, nbr_CantFrases_Train))
print("nbr_CantImag_Test:%d  nbr_CantFrases_Test:%d" % (nbr_CantImag_Test, nbr_CantFrases_Test))

GuardarPickle(tensor_Vocabulario, '_vocab.pickle')
GuardarPickle((arr_Imagenes_Train), '_image_train.pickle')
GuardarPickle((arr_Imagenes_Test), '_image_test.pickle')
GuardarPickle((nbr_CantFrases_Train, nbr_CantFrases_Test, nbr_Cant_Frases_X_Imag, nbr_CantImag_Train, nbr_CantImag_Test), '_n.pickle')
GuardarPickle((arr_Id_Frases_train, arr_Id_Frases_test), '_caption.pickle')
print(arr_Id_Frases_train.shape)
