import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


###### INITIALISATION #######

image_path = 'papillon.png'

image = mpimg.imread(image_path)
if image.dtype != np.uint8:  # Si les valeurs sont en flottant (0 à 1)
    image = (image * 255).astype(np.uint8)  # Convertir en entiers (0 à 255)


hauteur, largeur = image.shape[:2]
nouvelle_hauteur = (hauteur // 8) * 8
nouvelle_largeur = (largeur // 8) * 8

image = image[:,:,0]  #on garde qu'une seule couleur (2D)


# Tronquer l'image
image = image[:nouvelle_hauteur, :nouvelle_largeur]
taille_image = image.shape
#print(image)
# plt.imshow(image)
# plt.show()
#print(image.shape)


def division_blocs(image,taille):
    blocs = []
    for i in range(0, image.shape[0], taille):
        for j in range(0, image.shape[1], taille):
            bloc = image[i:i+taille, j:j+taille]
            blocs.append(bloc)
    return blocs

def reformer_image(blocs, taille_image, taille_bloc):
    blocs_par_ligne = taille_image[1] // taille_bloc
    image_recomposee = np.zeros(taille_image, dtype=blocs[0].dtype)

    index = 0
    for i in range(0, taille_image[0], taille_bloc):
        for j in range(0, taille_image[1], taille_bloc):
            image_recomposee[i:i+taille_bloc, j:j+taille_bloc] = blocs[index]
            index += 1

    return image_recomposee

blocs = division_blocs(image,8)
# print("bloc : ",blocs[0])
# plt.imshow(blocs[0])
# plt.show()

img = reformer_image(blocs,taille_image,8)
plt.imsave('image_recomposee.png',img)
image_recomposee = mpimg.imread('image_recomposee.png')
plt.imshow(image_recomposee)
plt.show()


#initialisation de P (matrice de passage) avec la formule de la double somme
P=np.zeros((8,8))
for i in range (8):
    for j in range (8):
        if i==0:
            ck=1/math.sqrt(2)
        else:
            ck=1
        P[i,j]= (1/2)*ck*math.cos(((2*j+1)*i*math.pi)/16)
#print(P)


####### COMPRESSION #########


# initialisation de Q : matrice de quantification Q dans la norme de compression JPEG
Q=np.array([[16,11,10,16,24,40,51,61],
            [12,12,13,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [72,92,95,98,112,100,103,99]])



def compression(blocs,P,Q):
    blocs_compressee = []
    boucle = 0
    for M in blocs :
        # boucle+=1
        # print(len(blocs),boucle)
        # print(M)
        D = P @ M @ P.T
        D=np.divide(D,Q)
        #print(D)
        # prendre la partie entière
        D = np.floor(D)
        blocs_compressee.append(D)
        
        # print("MATRICE D : \n",D)
        # plt.imsave('image_compresse.png',D)
        # image_compresse = mpimg.imread('image_compresse.png')
        # plt.imshow(image_compresse)
        # plt.show()
    return blocs_compressee

blocs_compressed = compression(blocs,P,Q)

img2 = reformer_image(blocs_compressed,taille_image,8)
print(img2)
#  — Compter le nombre de cœfficients non nuls pour obtenir le taux de compression
nb_coeff_non_zero = np.count_nonzero(img2)  # Nombre de coefficients non nuls
taux_compression = (nb_coeff_non_zero / (taille_image[1]*taille_image[0])) * 100  
print(f"taux de compression : {taux_compression}")


plt.imsave('image_recomposee2.png',img2)
image_recomposee2 = mpimg.imread('image_recomposee2.png')
plt.imshow(image_recomposee2)
plt.show()


def decompression(blocs,P,Q):
    blocs_decompressed = []
    for D in blocs:
        # Fonction decompression pour un bloc de l'image.
        # Initialisation de M la matrice decompressé.
        M = np.zeros((8,8))
        # Calcul de D_tilde la matrice D non divisé par Q (terme à terme).
        D_tilde = D * Q
        # Calcul de la transposée de la matrice de passage.
        P_transposee= np.transpose(P)
        # Calcul de la matrice décompressée.
        M = P_transposee @ D_tilde@ P
        blocs_decompressed.append(M)
    return blocs_decompressed


blocs_decompressed = decompression(blocs_compressed,P,Q)

img3 = reformer_image(blocs_decompressed,taille_image,8)
# print(img2)
plt.imsave('image_recomposee3.png',img3)
image_recomposee3 = mpimg.imread('image_recomposee3.png')
plt.imshow(image_recomposee3)
plt.show()
