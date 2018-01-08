#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>
using namespace cv;
using namespace std;

void picshadowx (char *srcPath);
void thresholdIntegral (Mat inputMat, Mat& outputMat);