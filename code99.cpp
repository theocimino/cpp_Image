//Bonjour Theo j'espère que tu vas bien 
/*pour executer ce code j'ai utiliser google colab (j'ai baisser les bras pour l'installation+ on a pas le temps ) et j'ai importer les fichiers main.cpp, matblotlibcpp.h et l'image 
  après j'ai fait la commande >>!g++ -std=c++11 main.cpp -I/usr/include/eigen3 -I/usr/include/python3.10 -lpython3.10 `pkg-config --cflags --libs opencv4` -o prog
>>!./prog*/
// tkt j'ai utiliser ctr+A
#include <iostream>
#include <cmath>///equivalent de math
#include <vector>
#include <Eigen/Dense>////equivalent de numpy
#include "matplotlibcpp.h" /// matplotlib
#include <opencv2/opencv.hpp>/// equivalent de matplotlib.image


namespace plt=matplotlibcpp;
using namespace std;
using namespace cv;
using namespace Eigen;

class CompImage{
    private:
    MatrixXd P;
    MatrixXi Q;
    int blocSize;
    public:
    
    //la construction des deux matrices Q et P
    void matriceP() {
        P = MatrixXd(blocSize, blocSize);
        for (int i = 0; i < blocSize; i++) {
            for (int j = 0; j < blocSize; j++) {
                double ck = (i == 0) ? 1.0 / sqrt(2) : 1.0;//if (i==0) donc ck=1.0 / sqrt(2) sinon ck=1
                P(i, j) = 0.5 * ck * cos(((2 * j + 1) * i * M_PI) / 16.0);
            }
        }
    }

    void matriceQ() {
        Q = MatrixXi(8, 8);
        Q << 16,11,10,16,24,40,51,61,
             12,12,13,19,26,58,60,55,
             14,13,16,24,40,57,69,56,
             14,17,22,29,51,87,80,62,
             18,22,37,56,68,109,103,77,
             24,35,55,64,81,104,113,92,
             49,64,78,87,103,121,120,101,
             72,92,95,98,112,100,103,99;
    }

    CompImage(int size = 8) : blocSize(size) {
        matriceP();
        matriceQ();
    }

     vector<MatrixXd> divideBlocks(const Mat& img) {
        vector<MatrixXd> blocs;
        int rows = img.rows - img.rows % blocSize;
        int cols = img.cols - img.cols % blocSize;

        for (int i = 0; i < rows; i += blocSize) {
            for (int j = 0; j < cols; j += blocSize) {
                MatrixXd bloc(blocSize, blocSize);
                for (int x = 0; x < blocSize; x++) {
                    for (int y = 0; y < blocSize; y++) {
                        bloc(x, y) = (double) img.at<uchar>(i + x, j + y);// uchar alias= unsigned char type entier non signé sur 8 bits peut prendre des valeurs de 0 à 255.
                    }
                }
                blocs.push_back(bloc);
            }
        }
        return blocs;
    }
        Mat reformImage(const vector<MatrixXd>& blocs, int rows, int cols) {
        Mat img_recom(rows, cols, CV_8UC1);//Les images couleur (CV_8UC3) utilisent trois uchar par pixel (R, G, B).
        int index = 0;
        for (int i = 0; i < rows; i += blocSize) {
            for (int j = 0; j < cols; j += blocSize) {
                const MatrixXd& bloc = blocs[index++];
                for (int x = 0; x < blocSize; x++)
                    for (int y = 0; y < blocSize; y++)
                        img_recom.at<uchar>(i + x, j + y) = (uchar)round(bloc(x, y));
            }
        }
        return img_recom;
    }
        vector<MatrixXd> compress(const vector<MatrixXd>& blocs) {
        vector<MatrixXd> compressed;
        for (const auto& M : blocs) {
            MatrixXd D = P * M * P.transpose();
            for (int i = 0; i < blocSize; i++)
                for (int j = 0; j < blocSize; j++)
                    D(i,j) = floor(D(i,j) / Q(i,j));
            compressed.push_back(D);
        }
        return compressed;
    }

};





int main() {
    string image_path = "MISSU.jpg";
    Mat image_origine= imread(image_path);
    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    imwrite("image_origine.png",image_origine);
    if (image.empty()) {
        cerr << "Erreur : Impossible de charger l'image !" << endl;
        return -1;
    }

    CompImage jpg;
    vector<MatrixXd> blocs = jpg.divideBlocks(image);
    vector<MatrixXd> blocs_compressed = jpg.compress(blocs);
    

    Mat img_recom = jpg.reformImage(blocs_compressed, image.rows - image.rows % 8, image.cols - image.cols % 8);
    imwrite("image_recomposee_cpp.png", img_recom);//sauvgarder l'image dans le fichier "image_recomposee_cpp.png"
    //imshow("Image Recomposee", img_recom);
    //waitKey(0);
    
    return 0;
}
