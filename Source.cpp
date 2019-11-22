#include <opencv2/core/utility.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>


/*********************************************
Improvements suggestions:

	1) Detect ALL faces and enlarge all
	2) Enlarge the eyse also
	3) 


***********************************************/

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



// bilinearInterpolation
// pre: Imgage& input, double row, and double col are passed in. 
//		input image is correctly allocated and formatted.
// post: perfroms bilinear interpolation for a target pixel of given coordinate
//		 row and col, and outputs a new pixel value. if any of four pixels around 
//		 the center pixel is out of the scope of input image, it returns a black pixel.
Vec3b bilinearInterpolation(const Mat& input, double row, double col) {

	Vec3b px;
	px[2] = 0.0;
	px[1] = 0.0;
	px[0] = 0.0;

	// there exists four pixels around the target pixel in the scope of the input image
	if (row >= 0.0 && row + 1.0 < input.rows &&
		col >= 0.0 && col + 1.0 < input.cols) {

		double redI, greenI, blueI;
		double alpha = row - floor(row);
		double beta = col - floor(col);

		// four pixels around the target pixel
		Vec3b pixelA = input.at<Vec3b>(row, col);
		Vec3b pixelB = input.at<Vec3b>(row + 1.0, col);
		Vec3b pixelC = input.at<Vec3b>(row, col + 1.0);
		Vec3b pixelD = input.at<Vec3b>(row + 1.0, col + 1.0);

		// red intensity
		redI = (1.0 - alpha) * (1.0 - beta) * pixelA[2]
			+ alpha * (1.0 - beta) * pixelB[2]
			+ (1.0 - alpha) * beta * pixelC[2]
			+ alpha * beta * pixelD[2];

		// green intensity
		greenI = (1.0 - alpha) * (1.0 - beta) * pixelA[1]
			+ alpha * (1.0 - beta) * pixelB[1]
			+ (1.0 - alpha) * beta * pixelC[1]
			+ alpha * beta * pixelD[1];

		// blue intensity
		blueI = (1.0 - alpha) * (1.0 - beta) * pixelA[0]
			+ alpha * (1.0 - beta) * pixelB[0]
			+ (1.0 - alpha) * beta * pixelC[0]
			+ alpha * beta * pixelD[0];

		// set new pixel values calculated
		px[2] = static_cast<int>(floor(redI));
		px[1] = static_cast<int>(floor(greenI));
		px[0] = static_cast<int>(floor(blueI));
	}
	return px;
}


// scale
// pre:	Image& input, double sx, and double sy are passed in. 
//		input image passed in is correctly allocated and formatted. 
// post: returns output image which is the width of the input that is 
//		 scaled by float sx and the height of the input scaled by float sy,
//		 using bilinear interpolation. pixels out side of the boundaries of
//		 the input image is represented as black.
Mat scale(const Mat& input, double sx, double sy) {
	Mat output(input);

	double centerRow = input.rows / 2.0;
	double centerCol = input.cols / 2.0;

	for (int row = 0; row < input.rows; row++) {
		for (int col = 0; col < input.cols; col++) {
			double x = (col - centerCol) / sy + centerCol;
			double y = (row - centerRow) / sx + centerRow;

			if (x > 0 && x < input.cols && y > 0 && y < input.rows) {
				output.at<Vec3b>(row, col) = bilinearInterpolation(input, y, x);
			}
		}
	}
	return output;
}


int main(int argc, const char** argv) {
	std::cout << "runs";

	Mat original = imread("OGJ.jpg");

	// testing scale function

	//Mat frame = scale(original, 1.2, 1.2);
	//imwrite("faceEnlarged", frame);
	//namedWindow("Capture - Face detection", WINDOW_NORMAL);
	////resizeWindow("Capture - Face detection", frame.cols / 6, frame.rows / 6);
	//imshow("Capture - Face detection", frame);
	//waitKey(0);

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(1)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(2)Error loading\n"); return -1; };
	//if (!smile_cascade.load(smile_cascade_name)) { printf("--(3)Error loading\n"); return -1; };

	//-- 2. Apply the classifier to the frame
	detectAndDisplay(original);
	cout << "Done";
	//imwrite("output.jpg", input);

	

	return 0;
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
	Mat enlargedJer;
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
	cout << "Number of faces detected : " << faces.size();
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

		enlargedJer = scale(jerFace, 2.0, 2.0);

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

	imwrite("enlargedFace.jpg", enlargedJer);
	/*namedWindow("Capture - enlargedFace", WINDOW_NORMAL);
	resizeWindow("Capture - enlargedFace", enlargedJer.cols / 6, enlargedJer.rows / 6);
	imshow("Capture - enlargedFace", enlargedJer);
	waitKey(0);*/

}


