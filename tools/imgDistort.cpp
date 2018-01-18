//paper:
//"Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis"
//Patrice etc. 2003
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>
using namespace cv;
using namespace std;
Mat elasticDistort(Mat& img,int gaussianWindowSize,double gaussianKernelSize)  
{  
    cvtColor(img,img,CV_RGB2GRAY);  
    Mat Xmv(img.rows,img.cols,CV_32FC1);  
    Mat Ymv(img.rows,img.cols,CV_32FC1);  
    //srand(time(NULL));  
    for(int i = 0; i<img.rows; i++)  
    {  
        float* p = Xmv.ptr<float>(i);  
        for(int j = 0; j<img.cols; j++)  
        {  
            p[j] = (rand() % 3 - 1) * 10;
        }  
    }  
    //srand(time(NULL) + 100);  
    srand(clock());  
    for(int i = 0; i<img.rows; i++)  
    {  
        float* p = Ymv.ptr<float>(i);  
        for(int j = 0; j<img.cols; j++)  
        {  
            p[j] = (rand() % 3 - 1) * 10;
        }  
    }  
    GaussianBlur(Xmv,Xmv,Size(gaussianWindowSize,gaussianWindowSize),gaussianKernelSize,gaussianKernelSize);  
    GaussianBlur(Ymv,Ymv,Size(gaussianWindowSize,gaussianWindowSize),gaussianKernelSize,gaussianKernelSize);  
  
      
    Mat result(img.rows,img.cols,img.type());  
    result.setTo(255);  
    for(int i = 0; i<img.rows; i++)  
    {  
        for(int j = 0; j<img.cols; j++)//这里进行了更改！  
        {     
            int newX_low = i + cvFloor(Xmv.at<float>(i,j));  
            int newX_high = i + cvCeil(Xmv.at<float>(i,j));  
            int newY_low = j + cvFloor(Ymv.at<float>(i,j));  
            int newY_high = j + cvCeil(Ymv.at<float>(i,j));  
  
  
            newX_low = newX_low < 0 ? 0 : newX_low;  
            newX_high = newX_high < 0 ? 0 : newX_high;  
            newY_low = newY_low < 0 ? 0 : newY_low;  
            newY_high = newY_high < 0 ? 0 : newY_high;  
  
  
            newX_low = newX_low >= img.rows ? img.rows - 1 : newX_low;  
            newX_high = newX_high >= img.rows ? img.rows - 1 : newX_high;  
            newY_low = newY_low >= img.cols ? img.cols - 1 : newY_low;  
            newY_high = newY_high >= img.cols ? img.cols - 1 : newY_high;  
  
  
            int sum = img.at<uchar>(newX_low,newY_low) + img.at<uchar>(newX_low,newY_high)+  
                         img.at<uchar>(newX_high,newY_low) + img.at<uchar>(newX_high,newY_high);  
            double avg = sum * 1.0 * 0.25;  
            if(avg < 110)  
            {  
                avg = 0;  
            }  
            else if(avg > 145)  
                avg = 255;  
            result.at<uchar>(i,j) = avg;  
            //result.at<uchar>(newX,newY) = img.at<uchar>(i,j);  
        }  
    }  
    //cout<<result<<endl;  
    return result;  
}  

int main(int argc, char** argv) {
    Mat img = imread(argv[1]);
    Mat ret = elasticDistort(img, 105, 4);
    imwrite("ret.bmp", ret);
    return 0;
}
