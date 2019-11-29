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
Mat detectAndDisplay(const Mat& frame);
void displayImage(const Mat& img, String winName);

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
//String smile_cascade_name = "haarcascade_smile.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
//CascadeClassifier smile_cascade;

Vec3b borderCheck(const Mat& input, int row, int col)
{
	int r, c;

	if (row < 0)
		r = 0;
	else if (row >= input.rows)
		r = input.rows- 1;
	else
		r = row;

	if (col < 0)
		c = 0;
	else if (col >= input.cols)
		c = input.cols - 1;
	else
		c = col;

	return input.at<Vec3b>(r, c);

}

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


//convolute
//Pre-condition:	input is a valid Image object
//					kernel is a valid Image object
//Post-condition:	returns an Image object resulted in convoluting
//					the kernel onto the input Image
Vec3b convolute(const Mat& input, const Mat& kernel, int oRow, int oCol)
{
	int cr = kernel.rows / 2;
	int cc = kernel.cols / 2;
	double totalRed = 0;
	double totalGreen = 0;
	double totalBlue = 0;

	// loop through kernel
	for (int i = -cr; i <= cr; i++)
	{
		for (int j = -cc; j <= cc; j++)
		{
			double pKernel = kernel.at<double>(cr + i, cc + j);
			Vec3b pImage = borderCheck(input, oRow + i, oCol + j);

			double newRed = pKernel * pImage[2];
			double newGreen = pKernel * pImage[1];
			double newBlue = pKernel * pImage[0];

			totalRed += newRed;
			totalGreen += newGreen;
			totalBlue += newBlue;
		}
	}

	Vec3b p;
	p[2] = totalRed;
	p[1] = totalGreen;
	p[0] = totalBlue;

	return p;
}

// Blurring the corner of the enlarged face
// 
// Pre-condition:
// Post-condition:
Mat blurCorner(const Mat& input, int width, int height, Point center) {
	Mat output = input.clone(); 
	

	//Make blur kernel
	//double kernelVals[9] = { 0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625 };
	double kernelVals[25] = {
	1.0 / 256, 4.0 / 256, 6.0 / 256, 4.0 / 256, 1.0 / 256,
	4.0 / 256, 16.0 / 256, 24.0 / 256, 16.0 / 256, 4.0 / 256,
	6.0 / 256, 24.0 / 256, 36.0 / 256, 24.0 / 256, 6.0 / 256,
	4.0 / 256, 16.0 / 256, 24.0 / 256, 16.0 / 256, 4.0 / 256,
	1.0 / 256, 4.0 / 256, 6.0 / 256, 4.0 / 256, 1.0 / 256
	};
	Mat blurKernel = Mat(5, 5, DataType<double>::type, kernelVals);

	for (int row = 0; row < output.rows; row++) {
		for (int col = 0; col < output.cols; col++) {
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

			//if at border blur
			if (LHS < 1) {
				Vec3b p = convolute(input, blurKernel, row, col);
				output.at<Vec3b>(row, col) = p;
			}
			
		}
	}
	displayImage(output, "bruh");
	return output;
}

// scale
// pre:	Image& input, double sx, and double sy are passed in. 
//		input image passed in is correctly allocated and formatted. 
// post: returns output image which is the width of the input that is 
//		 scaled by float sx and the height of the input scaled by float sy,
//		 using bilinear interpolation. pixels out side of the boundaries of
//		 the input image is represented as black.
Mat scale(const Mat& input, int cx, int cy, double s) {
	Mat output = input.clone();//(input.rows, input.cols, CV_8UC3);

	double centerRow = cy;
	double centerCol = cx;

	for (int row = 0; row < input.rows; row++) {
		for (int col = 0; col < input.cols; col++) {
			double x = (col - centerCol) / s + centerCol;
			double y = (row - centerRow) / s + centerRow;

			if (x > 0 && x < input.cols && y > 0 && y < input.rows) {
				output.at<Vec3b>(row, col) = input.at<Vec3b>(y, x); //bilinearInterpolation(input, y, x);
			}
		}
	}

	return output;
}

// Overlay cropped enlarged head onto original picture
//
// Pre-condition: - sourceImage will be the enlarged head
//				  - originalImage will be the original image
// Post-condition: returns the overlayed image, without changing input objects
Mat overlay(const Mat& face, const Mat& input) {
	
	Mat output = input.clone();
	
	Vec3b green = { 0, 255, 0 };

	for (int col = 0; col < input.cols; col++) {
		for (int row = 0; row < input.rows; row++) { 
			
			// if the color is not green, then grab it
			if (face.at<Vec3b>(row, col) != green)
				output.at<Vec3b>(row, col) = face.at<Vec3b>(row, col);
			
		}
	}

	//displayImage(output, "Overlay");
	imwrite("Overlay.jpg", output);

	return output;
}


// FetchFace
// ---------
// Pre-condition: input is a valid image
// Post-condition: crops out ONE face and puts it onto a green background
// ---------
//		- const Mat& input: the source image
//		- int width: the widht of the face
//		- int height: the height of the face (chin to hairtop)
//		- Point center: the center point of the face (around the nose)
Mat fetchFace(const Mat& input, int width, int height, Point center) {

	Mat onlyFace = input.clone();

	for (int row = 0; row < onlyFace.rows; row++) {
		for (int col = 0; col < onlyFace.cols; col++) {

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
			
			// makes it green if outside the head
			if (LHS > 1) 
				onlyFace.at<Vec3b>(row, col) = { 0, 255, 0 }; 
		}
	}
	return onlyFace;

}

/** @function detectAndDisplay */
Mat detectAndDisplay(const Mat& frame)
{
	double scaleFactor = 2.0;
	Point finalCenter;
	Mat jerFace;
	Mat enlargedJer;
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);


	//-- Detect faces
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces);

	for (auto val : faces)
		cout << val << " ";

	cout << "Number of faces detected : " << faces.size();
	cout << endl;

	for (size_t i = 0; i < faces.size(); i++)
	{
		int cx = faces[i].x + faces[i].width / 2;
		int cy = faces[i].y + faces[i].height / 2 - faces[i].height * 0.08;
		
		// Point object of the face's center
		Point center(cx, cy);
		finalCenter = center;

		//ellipse(frame, center, Size(faces[i].width / 2, (faces[i].height / 2) * 1.3), 0, 0, 360, Scalar(255, 0, 255), 8);


		Point topLeft(faces[i].x, faces[i].y);
		cout << topLeft << endl;
		Point bottomRight(faces[i].x + faces[i].width, faces[i].y + faces[i].height);

		// rectangle(frame, topLeft, bottomRight, Scalar(255, 0, 255), 8, 0, 0);

		Mat faceROI = frame_gray(faces[i]);

		//-- In each face, detect eyes
		std::vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes);

		// Circling the eye, do we need this?
		//for (size_t j = 0; j < eyes.size(); j++)
		//{
		//	Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
		//	int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
		//	//circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
		//}

		// An image with the cropped face with a green background
		jerFace = fetchFace(frame, faces[i].width, faces[i].height * 1.3, center);
		
		// An image with the enlarged cropped face with a green background
		enlargedJer = scale(jerFace, cx, cy, scaleFactor);
	}

	//displayImage(jerFace, "Capture - Face detection");

	//imwrite("enlargedFace.jpg", enlargedJer);

	/*Mat blurredJer = blurCorner(enlargedJer, enlargedJer.cols, enlargedJer.rows, finalCenter);
	namedWindow("Blurred Face", WINDOW_NORMAL);
	resizeWindow("Blurred Face", blurredJer.cols / 6, blurredJer.rows / 6);
	imshow("Blurred Face", blurredJer);
	waitKey(0);

	imwrite("blurredFace.jpg", blurredJer);*/
	Mat overlayed = overlay(enlargedJer, frame);

	imwrite("beforeBlur.jpg", overlayed);
	Mat output = blurCorner(overlayed, faces[0].width * scaleFactor, faces[0].height * 1.3 * scaleFactor, finalCenter);

	//displayImage(output, "blurred Corners");
	imwrite("afterBlur.jpg", output);
	return enlargedJer;
}

// displayImage
//
// Pre-condition: img is a valid img
// Post-condition: Displays the image onto a popUp window
void displayImage(const Mat& img, String winName)
{
	namedWindow(winName, WINDOW_NORMAL);
	resizeWindow(winName, img.cols / 6, img.rows / 6);
	imshow(winName, img);
	waitKey(0);
}


int main(int argc, const char** argv) {
	std::cout << "runs";

	Mat original = imread("OGJ.jpg");
	
	// testing scale function

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(1)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(2)Error loading\n"); return -1; };
	//if (!smile_cascade.load(smile_cascade_name)) { printf("--(3)Error loading\n"); return -1; };

	//-- 2. Apply the classifier to the frame
	imwrite("original.jpg", original);
	Mat enlarged = detectAndDisplay(original); //this changed original fo some reason 
	imwrite("originalAfterEnlarged.jpg", original);
	
	//Mat ans = detectAndDisplay(original);


	//Mat overlayed = overlay(enlarged, original);
	//imwrite("originalAfterOverlay.jpg", original);
	cout << "Done";
	//imwrite("output.jpg", input);


	return 0;
}

