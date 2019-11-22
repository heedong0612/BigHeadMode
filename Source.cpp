#include <opencv2/core/utility.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
//String smile_cascade_name = "haarcascade_smile.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
//CascadeClassifier smile_cascade;

int main(int argc, const char** argv) {
	std::cout << "runs";

	Mat frame = imread("OGJ.jpg");

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(1)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(2)Error loading\n"); return -1; };
	//if (!smile_cascade.load(smile_cascade_name)) { printf("--(3)Error loading\n"); return -1; };

	//-- 2. Apply the classifier to the frame
	detectAndDisplay(frame);
	cout << "Done";
	//imwrite("output.jpg", input);

	

	return 0;
}


Vec3b bilinearInterpolate(const Mat& input, double x, double y)
{
	Vec3b ans;
	ans[2] = 0;
	ans[1] = 0;
	ans[0] = 0;

	//check if coordinate is out of bounds in the input image
	if (x >= 0 && x < input.cols && y >= 0 && y < input.rows)
	{
		double x_f = floor(x);
		double y_f = floor(y);
		double alpha = y - y_f;
		double beta = x - x_f;

		Vec3b top_l = input.at< Vec3b>(y_f, x_f);
		Vec3b top_r = input.at< Vec3b>(y_f, x_f + 1); //row, col + 1
		Vec3b bot_l = input.at< Vec3b>(y_f + 1, x_f); //row + 1, col
		Vec3b bot_r = input.at< Vec3b>(y_f + 1, x_f + 1); //row + 1, col + 1

		int r = (1 - alpha) * (1 - beta) * top_l[2]
			+ alpha * (1 - beta) * bot_l[2]
			+ (1 - alpha) * beta * top_r[2]
			+ alpha * beta * bot_r[2];
		int g = (1 - alpha) * (1 - beta) * top_l[1]
			+ alpha * (1 - beta) * bot_l[1]
			+ (1 - alpha) * beta * top_r[1]
			+ alpha * beta * bot_r[1];
		int b = (1 - alpha) * (1 - beta) * top_l[0]
			+ alpha * (1 - beta) * bot_l[0]
			+ (1 - alpha) * beta * top_r[0]
			+ alpha * beta * bot_r[0];

		ans[2] = r;
		ans[1] = g;
		ans[0] = b;
	}
	return ans;
}

void scale(Mat& input, double sx, double sy)
{
	double cx = input.cols / 2;
	double cy = input.rows / 2;

	for (int row = 0; row < input.rows; row++)
	{
		for (int col = 0; col < input.cols; col++)
		{

			double newX = (col - cx) / sx + cx; //new x or new col
			double newY = (row - cy) / sy + cy; //new y or new row

			Vec3b pix = bilinearInterpolate(input, newX, newY);
			input.at<Vec3b>(row, col) = pix;
		}
	}
}

 Mat enlargeTheFace(const Mat& original, const Mat& onlyFace) {

	 Mat enlargedFace(onlyFace);

	 scale(enlargedFace, 2, 2);


	 imwrite("faceOnly.jpg", enlargedFace);
	 namedWindow("Capture - Face detection", WINDOW_NORMAL);
	 resizeWindow("Capture - Face detection", enlargedFace.cols / 6, enlargedFace.rows / 6);
	 imshow("Capture - Face detection", enlargedFace);
	 waitKey(0);

	 return enlargedFace;
 }


Mat fetchFace(const Mat& original, int startingX, int startingY, int width, int height, Point center) {

	Mat onlyFace(original);

	for (int row = 0; row < original.rows; row++) {
		for (int col = 0; col < original.cols; col++) {

			// if in the elipse region
			// do nothing
			// if not, put in green

			// ellipse equation
			// (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1
			// x is col
			// y is row
			// h, k is the ellipse x, y center
			// a is the x-axis radius
			// b is the y-axis radius

			
			double leftHandSide1 = pow((col - center.x), 2) / pow((width / 2), 2);
			double leftHandSide2 = pow((row - center.y), 2) / pow((height / 2), 2);
			double LHS = leftHandSide1 + leftHandSide2;

			if (LHS > 1) {
				//cout << "in ";
				onlyFace.at<Vec3b>(row, col) = { 0, 255, 0 };
			}

		}
	}
	return onlyFace;

}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	Mat jerFace;
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);


	//-- Detect faces
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces);

	for(auto val : faces)
	{
		cout << val << " ";
	}
	cout << "len of face : " << faces.size();
	cout << endl;

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2 - faces[i].height * 0.08);
		ellipse(frame, center, Size(faces[i].width / 2, (faces[i].height / 2)*1.3), 0, 0, 360, Scalar(255, 0, 255), 8);

	
		Point topLeft(faces[i].x, faces[i].y);
		cout << topLeft << endl;
		Point bottomRight(faces[i].x + faces[i].width, faces[i].y + faces[i].height);

		// rectangle(frame, topLeft, bottomRight, Scalar(255, 0, 255), 8, 0, 0);

		Mat faceROI = frame_gray(faces[i]);

		//-- In each face, detect eyes
		std::vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes);
		
		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
		}

		jerFace = fetchFace(frame, faces[i].x, faces[i].y, faces[i].width, faces[i].height * 1.3, center);

		enlargeTheFace(jerFace, jerFace);

		// Detect smile
	/*	std::vector<Rect> smile;
		smile_cascade.detectMultiScale(faceROI, smile, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(60, 60));

		for (size_t j = 0; j < smile.size(); j++) {
			cv::Point center(faces[i].x + smile[j].x + smile[j].width * 0.5, faces[i].y + smile[j].y + smile[j].height * 0.5);
			cv::ellipse(frame, center, cv::Size(smile[j].width * 0.5, smile[j].height * 0.5), 0, 0, 360, cv::Scalar(0, 255, 255), 4, 8, 0);
		}*/

	}




	//-- Show what you got
	//imwrite("output.jpg", frame);
	//namedWindow("Capture - Face detection", WINDOW_NORMAL);
	//resizeWindow("Capture - Face detection", frame.cols / 6, frame.rows / 6);
	//imshow("Capture - Face detection", frame);
	//waitKey(0);


	imwrite("faceOnly.jpg", jerFace);
	namedWindow("Capture - Face detection", WINDOW_NORMAL);
	resizeWindow("Capture - Face detection", jerFace.cols / 6, jerFace.rows / 6);
	imshow("Capture - Face detection", jerFace);
	waitKey(0);

}


