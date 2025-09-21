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

    
    vector<MatrixXd> division_blocs(const Mat& img){
        vector<MatrixXd> blocs;
        int rows = img.rows - img.rows % blocSize;
        int cols = img.cols - img.cols % blocSize;

        for (int i= 0 ;i <rows;i += blocSize ){
            for( int j=0; j<cols;j+=blocSize){
                MatrixXd bloc(blocSize, blocSize);
                for (int x = 0; x < blocSize; x++) {
                    for (int y = 0; y < blocSize; y++) {
                        bloc(x, y) = (double) img.at<uchar>(i + x, j + y);//
                    }
                }
                blocs.push_back(bloc);// blocs.append(bloc)

            }
        }
        return blocs;
    }
    
        Mat reformer_image(const vector<MatrixXd>& blocs, int rows, int cols) {
        Mat img_reformee(rows, cols, CV_8UC1);//image finale vide avant le remplissage
        int index = 0;// compteur de blocs
        for (int i = 0; i < rows; i += blocSize) {
            for (int j = 0; j < cols; j += blocSize) {
                const MatrixXd& bloc = blocs[index++];
                for (int x = 0; x < blocSize; x++)
                    for (int y = 0; y < blocSize; y++)
                        img_reformee.at<uchar>(i + x, j + y) = (uchar)round(bloc(x, y));
            }
        }
        return img_reformee;
    }
        vector<MatrixXd> compress(const vector<MatrixXd>& blocs) {
        vector<MatrixXd> compressed;
        for (int k = 0; k < blocs.size(); k++) {
            MatrixXd M = blocs[k];
            MatrixXd D = P * M * P.transpose();
            for (int i = 0; i < blocSize; i++)
                for (int j = 0; j < blocSize; j++)
                    D(i,j) = floor(D(i,j) / Q(i,j));//prendre la partie entière
            compressed.push_back(D);
        }
        return compressed;
    }
    double Taux_Comp(const Mat& img){
        int nb_coeff_non_zero=countNonZero(img);
        int total_pixels = img.rows * img.cols;
        double taux_compression = (double)nb_coeff_non_zero / total_pixels * 100;
        cout << "taux de compression : " << taux_compression << "%" << endl;
        return taux_compression;
    }
    vector<MatrixXd> decompress(const vector<MatrixXd>& blocs_compressed) {
        vector<MatrixXd> decompressed;
        for (int k = 0; k < blocs_compressed.size(); k++){
            MatrixXd D_new = blocs_compressed[k];
            for (int i = 0; i < blocSize; i++){
                for (int j = 0; j < blocSize; j++){
                    D_new(i,j) *= Q(i,j);
                }  
            }      
            MatrixXd M = P.transpose() * D_new * P;
            decompressed.push_back(M);
        }    
        return decompressed;
    }


};






int main() {
    string image_path = "MISSU.jpg";
    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    imwrite("image_origine.png",image);
    if (image.empty()) {
        cerr << "Erreur : Impossible de charger l'image !" << endl;
        return -1;
    }

    CompImage jpg;
    vector<MatrixXd> blocs = jpg.division_blocs(image);
    vector<MatrixXd> blocs_compressed = jpg.compress(blocs);
    vector<MatrixXd> blocs_decompressed = jpg.decompress(blocs_compressed);

    Mat img_compresee = jpg.reformer_image(blocs_compressed, image.rows - image.rows % 8, image.cols - image.cols % 8);
    Mat img_reformee = jpg.reformer_image(blocs_decompressed, image.rows - image.rows % 8, image.cols - image.cols % 8);

    imwrite("image_composee_cpp.png", img_compresee);
    imwrite("image_recomposee_cpp.png", img_reformee);
    jpg.Taux_Comp(img_reformee);

    return 0;
}
