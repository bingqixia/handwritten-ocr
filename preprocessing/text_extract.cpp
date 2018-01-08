//#include "stdafx.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//#define INPUT_FILE              "1.jpg"
#define OUTPUT_FOLDER_PATH      string("")

int main(int argc, char **argv)
{

    Mat large = imread(argv[1]);
    Mat rgb;
    // downsample and use it for processing
    pyrDown(large, rgb);
    Mat small;
    cvtColor(rgb, small, CV_BGR2GRAY);
    // morphological gradient
    Mat grad;
    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));//形态学滤波
    morphologyEx(small, grad, MORPH_GRADIENT, morphKernel);//凸显轮廓
    //imwrite("grad.png", grad);
    // binarize
    Mat bw;
    //threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
    threshold(grad, bw, 0.0, 255.0,  THRESH_BINARY|THRESH_OTSU);
    //imwrite("black.png", bw);
    // connect horizontally oriented regions
    Mat connected;
    morphKernel = getStructuringElement(MORPH_RECT, Size(3, 1));
    morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
    // find contours
    Mat mask = Mat::zeros(bw.size(), CV_8UC1);
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    Mat dst = Mat::zeros(connected.rows, connected.cols, CV_8UC3);

    FILE *fp = fopen("../Train/path_info.txt", "w");

    // filter contours
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
        double area = contourArea(contours[idx]);
        //面积小于1000的全部筛选掉
        if (area < 200)
            continue;
        Rect rect = boundingRect(contours[idx]);
        vector<Rect>rects;
        Mat maskROI(mask, rect);
        maskROI = Scalar(0, 0, 0);

        // fill the contour
        drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
        // ratio of non-zero pixels in the filled region
        double r = (double)countNonZero(maskROI)/(rect.width*rect.height);
       if ((rect.width > 10) && (rect.height > 15)) {
            //rectangle(rgb, rect, Scalar(0, 255, 0), 2);
            Mat imageROI;
            char *path;
            rgb(rect).copyTo(imageROI);
            rects.push_back(rect);
            sprintf(path, "res/rect_%d.png", idx);
            fprintf(fp, "%s\n", path);
            imwrite(path, imageROI);
        }  
    }
    imwrite(OUTPUT_FOLDER_PATH + string("rgb.jpg"), rgb);

    return 0;
}
