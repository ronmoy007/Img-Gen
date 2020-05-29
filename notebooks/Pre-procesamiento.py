# Contamos únicamente con 8091 imágenes
import numpy as np
import pandas as pd
import cv2
from pathlib import Path


def GenerarListaImagenes(str_Path):

    np_Archivos = np.empty([0])
    for file in Path(str_Path+'Flicker8k_Dataset').glob('*'):
        # print(file)
        # list_Archivos.append(file.name)
        np_Archivos = np.append(np_Archivos, [file.name])

    # Cantidad de imágenes disponibles
    # len(np_Archivos)

    # Se construye el dataframe con base en el arreglo de campos
    columns = ['id_imagen']
    df = pd.DataFrame(data=np_Archivos, columns=columns)
    df.to_csv(str_Path+'/Preprocessing/lista_imagenes.csv', index=False, header=True)


def FormatearDescripciones(str_Path, str_Archiv_Origen, str_NombreCsv):

    # Recorremos el archivo para eliminar los tabs
    str_Fuente = str_Path + '/Flickr8k_text/' + str_Archiv_Origen
    str_Destino = str_Path + '/Flickr8k_text/' + 'Tmp_' + str_Archiv_Origen
    with open(str_Fuente) as Origen, open(str_Destino, 'w') as Destino:
        for line in Origen:
            Destino.write(line.replace('\t', ' '))

    # Recorremos línea a línea el archivo para encontrar en qué punto se separa
    # el id de imagen del texto
    np_Array = np.empty([0,3])
    with open(str_Destino) as file:
        for line in file:
            nbr_Index = line.find(' ')
            line = line.replace('\n', '')
            line = line.replace(' .', '')
            # print(nbr_Index)
            str_Substr1 = line[0:nbr_Index-2]
            str_Substr2 = line[nbr_Index-1]
            str_Substr3 = line[nbr_Index+1:]
            # print(str_Substr2)
            # list_IdImagenes.append(str_Substr1)
            # list_Descr.append(str_Substr2)
            np_Array = np.append(np_Array, [[str_Substr1, str_Substr2, str_Substr3]], axis=0)

    # Se construye el dataframe con base en el arreglo de campos
    columns = ['id_imagen', 'id_seqq', 'descr']
    df = pd.DataFrame(data=np_Array, columns=columns)
    df.to_csv(str_Path+'/Preprocessing/'+str_NombreCsv, index=False, header=True)


def resize_img(img_file, width, height):
    """Función para cambiar las dimensiones de una imagen

    Args:
        img_file: Archivo de imagen en formato jpg
        width: Anchura a la que se desea modificar la imagen
        height: Altura a la que se desea modificar la imagen

    Returns:
        resized: Arreglo de vectores que representan a la imagen con sus nuevas
            dimensiones
    """

    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    #print('Original Dimensions :', img.shape)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #print('Resized Dimensions :', resized.shape)
    return resized


str_Path = '/Users/Marco/github/Img-Gen/'
# GenerarListaImagenes(str_Path)
# FormatearDescripciones(str_Path, 'Flickr8k.token.txt', 'Token.csv')

# 40,000 descripciones
df_Tokens = pd.read_csv(str_Path+'/Preprocessing/Token.csv')
df_Imagenes = pd.read_csv(str_Path+'/Preprocessing/lista_imagenes.csv')

# Temas de embeddings:
# Colores: white, black, blue, yellow, red
# Especie/tipo: man, woman, girl, boy, child, baby, dog
df_Reducido = df_Tokens.loc[df_Tokens['id_seqq'] == 0]
df_Reducido.to_csv(str_Path+'/Preprocessing/Token_Reducido.csv', index=False, header=True)

# Animales
df_Dogs = df_Reducido.query('descr.str.contains("dog")')
df_Cats = df_Reducido.query('descr.str.contains("cat")')

# Adultos
df_Men = df_Reducido.query('descr.str.contains("man")')
df_Women = df_Reducido.query('descr.str.contains("woman")')

# Niños
df_Boys = df_Reducido.query('descr.str.contains("boy")')
df_Girls = df_Reducido.query('descr.str.contains("girl")')

# df_Dogs.to_csv(str_Path+'/Preprocessing/df_Dogs.csv', index=False, header=True)
# df_Cats.to_csv(str_Path+'/Preprocessing/df_Cats.csv', index=False, header=True)
# df_Men.to_csv(str_Path+'/Preprocessing/df_Men.csv', index=False, header=True)
# df_Women.to_csv(str_Path+'/Preprocessing/df_Women.csv', index=False, header=True)
# df_Boys.to_csv(str_Path+'/Preprocessing/df_Boys.csv', index=False, header=True)
# df_Girls.to_csv(str_Path+'/Preprocessing/df_Girls.csv', index=False, header=True)

# Colores de cosas
df_White = df_Reducido.query('descr.str.contains("white")')
df_Black = df_Reducido.query('descr.str.contains("black")')
df_Blue = df_Reducido.query('descr.str.contains("blue")')
df_Yellow = df_Reducido.query('descr.str.contains("yellow")')
df_Red = df_Reducido.query('descr.str.contains("red")')
df_Green = df_Reducido.query('descr.str.contains("green")')

# Colores de pelo
df_Brown = df_Reducido.query('descr.str.contains("brown")')
df_Blonde = df_Reducido.query('descr.str.contains("blonde")')

# df_White.to_csv(str_Path+'/Preprocessing/df_White.csv', index=False, header=True)
# df_Black.to_csv(str_Path+'/Preprocessing/df_Black.csv', index=False, header=True)
# df_Blue.to_csv(str_Path+'/Preprocessing/df_Blue.csv', index=False, header=True)
# df_Yellow.to_csv(str_Path+'/Preprocessing/df_Yellow.csv', index=False, header=True)
# df_Red.to_csv(str_Path+'/Preprocessing/df_Red.csv', index=False, header=True)
# df_Green.to_csv(str_Path+'/Preprocessing/df_Green.csv', index=False, header=True)
# df_Brown.to_csv(str_Path+'/Preprocessing/df_Brown.csv', index=False, header=True)
# df_Blonde.to_csv(str_Path+'/Preprocessing/df_Blonde.csv', index=False, header=True)
