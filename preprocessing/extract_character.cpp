#include "extract_character.h"
Mat picshadowy (Mat binary);
//积分二值化
void thresholdIntegral (Mat inputMat, Mat& outputMat)
{

    int nRows = inputMat.rows;
    int nCols = inputMat.cols;

    // create the integral image
    Mat sumMat;
    integral (inputMat, sumMat);

    int S = MAX (nRows, nCols) / 8;
    double T = 0.15;

    // perform thresholding
    int s2 = S / 2;
    int x1, y1, x2, y2, count, sum;

    int* p_y1, *p_y2;
    uchar* p_inputMat, *p_outputMat;

    for (int i = 0; i < nRows; ++i)
    {
        y1 = i - s2;
        y2 = i + s2;

        if (y1 < 0)
        {
            y1 = 0;
        }
        if (y2 >= nRows)
        {
            y2 = nRows - 1;
        }

        p_y1 = sumMat.ptr<int> (y1);
        p_y2 = sumMat.ptr<int> (y2);
        p_inputMat = inputMat.ptr<uchar> (i);
        p_outputMat = outputMat.ptr<uchar> (i);

        for (int j = 0; j < nCols; ++j)
        {
            // set the SxS region
            x1 = j - s2;
            x2 = j + s2;

            if (x1 < 0)
            {
                x1 = 0;
            }
            if (x2 >= nCols)
            {
                x2 = nCols - 1;
            }

            count = (x2 - x1)* (y2 - y1);

            // I(x,y)=s(x2,y2)-s(x1,y2)-s(x2,y1)+s(x1,x1)
            sum = p_y2[x2] - p_y1[x2] - p_y2[x1] + p_y1[x1];

            if ((int) (p_inputMat[j] * count) < (int) (sum* (1.0 - T)))
            {
                p_outputMat[j] = 0;
            }
            else
            {
                p_outputMat[j] = 255;
            }
        }
    }
}
//垂直方向投影
void picshadowx (char* srcPath)
{
    Mat src = cv::imread(srcPath);

    if (src.empty()) {
        cerr << "Problem loading image!!!" << endl;
        return;
    }
      
    Mat gray;

    if (src.channels() == 3) {
        cv::cvtColor (src, gray, CV_BGR2GRAY);       
    } else {
            gray = src;
    }
     
    Mat binary = Mat::zeros (gray.size(), CV_8UC1);
    thresholdIntegral (gray, binary);
    int* blackcout = new int[binary.cols];
    memset (blackcout, 0, binary.cols * 4);

    for (int i = 0; i < binary.rows; i++)
    {
        for (int j = 0; j < binary.cols; j++)
        {
            if (binary.at<uchar> (i, j) == 0)
            {
                blackcout[j]++; //垂直投影按列在x轴进行投影
            }
        }
    }

    vector<Mat> result;
    int startindex = 0;
    int endindex = 0;
    int char_c = 0, ipx = 0, sum_c = 0, now_c = 0;
    int arr_s[100], k = 0;
    memset(arr_s, 0, sizeof(arr_s));
    bool inblock = false, isFirst = true; //是否遍历到字符位置
    //Mat paintx (binary.size(), CV_8UC1, Scalar (255)); //创建一个全白图片，用作显示
   // printf("cols: %d %d\n", paintx.cols, binary.cols);
    for (int i = 0; i < binary.cols-1; i++) {
    
        if (!inblock && blackcout[i] > 5) //进入有字符区域
        {
            inblock = true;
            isFirst = false;
            startindex = i;
            char_c = result.size();
            if (char_c > 0) {
                ipx = sum_c / char_c;
               // printf("sum %d char_c %d\n", sum_c, char_c);
               // printf("ipx %d now_c %d\n", ipx, now_c);
                if(ipx < now_c && now_c-ipx > 20) {
                    arr_s[result.size()] = 1;
                    //printf("%d\n", result.size());
                    // printf("sum %d char_c %d\n", sum_c, char_c);
                    //printf("ipx %d now_c %d\n", ipx, now_c);
                }
            }
          //  cout << "startindey:" << startindex << endl;
        }

        if (inblock && blackcout[i] < 6) //进入浮动空白区
        {
            endindex = i;
            now_c = 0;
            inblock = false;
            Mat roi = binary.colRange(startindex, endindex+1); //从而记录从开始到结束行的位置，即可进行行切分
            result.push_back (roi);
        } else if (!isFirst && blackcout[i] == 0) {
            sum_c ++;
            now_c ++;
        }

    }
    FILE *fp = fopen("tmp/path_info_tmp.txt", "w");
    for (int i = 0; i < result.size(); i++)
    {
        if (arr_s[i])  fprintf(fp, "end\n");
        Mat tmp = result[i];
        if (tmp.cols > 30) {
            tmp = picshadowy(tmp);
            char path[50];
            sprintf(path, "tmp/testy_%d.bmp", i);
            //printf("%s\n", path);
            imwrite(path, tmp);
            fprintf(fp, "%s\n", path);

        }
      //  imshow ("testy"+to_string (i), tmp);
     }
    delete blackcout;
    fclose(fp);
}

//水平方向投影并行分割
Mat picshadowy (Mat binary)
{  
   //是否为白色或者黑色根据二值图像的处理得来
    Mat painty (binary.size(), CV_8UC1, Scalar (255)); //初始化为全白
   
   //水平投影
   int* pointcount = new int[binary.rows]; //在二值图片中记录行中特征点的个数
    memset (pointcount, 0, binary.rows * 4);//注意这里需要进行初始化

    for (int i = 0; i < binary.rows; i++)
    {
        for (int j = 0; j < binary.cols; j++)
        {
            if (binary.at<uchar> (i, j) == 0)
            {
                pointcount[i]++; //记录每行中黑色点的个数 //水平投影按行在y轴上的投影
            }
        }
    }

    for (int i = 0; i < binary.rows; i++) {
        for (int j = 0; j < pointcount[i]; j++) //根据每行中黑色点的个数，进行循环
        {
            
            painty.at<uchar> (i, j) = 0; 
        }

    }

   // imshow ("painty", painty);
  //  vector<Mat>result;
    int startindex = 0;
    int endindex = 0;
    bool inblock = false; //是否遍历到字符位置

    for (int i = 0; i < painty.rows; i++) {
      
        if (!inblock&&pointcount[i] > 5) //进入有字符区域
        {
            inblock = true;
            startindex = i;
            //cout << "startindex:" << startindex << endl;
        }
        if (inblock && pointcount[i] == 0) //进入浮动空白区
        {
            endindex = i;
            inblock = false;
            Mat roi = binary.rowRange(startindex, endindex+1); //从而记录从开始到结束行的位置，即可进行行切分
          //  result.push_back(roi);
            return roi;
        }
    }
    delete pointcount;
}

int main (int argc, char* argv[]) {
   
      Mat src = cv::imread(argv[1]);

        if (src.empty()) {
            cerr << "Problem loading image!!!" << endl;
            return -1;
        }
      
        Mat gray;

        if (src.channels() == 3) {
            cv::cvtColor (src, gray, CV_BGR2GRAY);       
        } else {
            gray = src;
        }
     
        Mat bw2 = Mat::zeros(gray.size(), CV_8UC1);
        thresholdIntegral(gray, bw2);
        picshadowx(argv[1]);
    return 0;
}