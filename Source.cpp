#include <opencv2/core/utility.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp" 
#include <iostream>

#define PI 3.14159265

/*********************************************
Improvements suggestions:
	1) Detect ALL faces and enlarge all
	2) Enlarge the eyse also
	3) tilted head

NEED to FIX:
	1) enlarging scale should depend on the picture size and face size ratio


MULTIPLE FACE DETECTION
	1) count the number of faces detected
	2) make a vector of that size to store face info (center, width, height) for ellipse
	3) for each face, enlarge and stitch
	4) go through the vectors and blur

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

// borderCheck
// Pre-condition: input is a valid Mat object
//					row and col can be any real integer
// Post-condition: checks whether the element at row, col is a valid location 
//					of the input Mat
Vec3b borderCheck(const Mat& input, int row, int col)
{
	int r, c;

	row < 0 ? r = 0 : row >= input.rows ? r = input.rows - 1 : r = row;
	col < 0 ? c = 0 : col >= input.cols ? c = input.cols - 1 : c = col;

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
//					Convolutes ONE patch of the image input with the location of oRow and oCol
//					with the kernel kernel.
//					Assumes kernel origin is always at the center
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

// calcEllipse
// Pre-condition : row, col are non-negative integers
// Post-condition : returns a value based on the ellipse mathematical equation
// ellipse equation with rotation
	// ((X-Cx)*cos⁡(θ)+(Y-Cy)*sin⁡(θ))^2/(Rx)^2 +
	// ((X-Cx)*sin⁡(θ)-(Y-Cy)*cos⁡(θ))^2/(Ry)^2 =1
	// (Cx,Cy)  is the center of the Ellipse.
 	// Rx is the Major-Radius, and Ry is the Minor-Radius.
	// θ is the angle of the Ellipse rotation.

double calcEllipse(int row, int col, Point center, int width, int height, double angle = 0)
{
	double leftHandSide1, leftHandSide2, ans;
	leftHandSide1 = pow((col - center.x) * cos(angle * PI / 180) + (row - center.y) * 
		sin(angle * PI / 180), 2) / pow((width/2), 2);
	leftHandSide2 = pow((col - center.x) * sin(angle * PI / 180) - (row - center.y) * 
		cos(angle * PI / 180), 2) / pow((height/2), 2);
	ans = leftHandSide1 + leftHandSide2;

	return ans;
}

// Blurring the corner of the enlarged face
// 
// Pre-condition:
// Post-condition:
Mat blurCorner(const Mat& input, int width, int height, Point center, double angle) {
	Mat output = input.clone();

	//Make blur kernel
	//double kernelVals[9] = { 0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625 };
	/* double kernelVals[25] = {
	1.0 / 256, 4.0 / 256, 6.0 / 256, 4.0 / 256, 1.0 / 256,
	4.0 / 256, 16.0 / 256, 24.0 / 256, 16.0 / 256, 4.0 / 256,
	6.0 / 256, 24.0 / 256, 36.0 / 256, 24.0 / 256, 6.0 / 256,
	4.0 / 256, 16.0 / 256, 24.0 / 256, 16.0 / 256, 4.0 / 256,
	1.0 / 256, 4.0 / 256, 6.0 / 256, 4.0 / 256, 1.0 / 256
	}; */

	double kernelVals[49] = { 0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036, 0.000363,
					0.003676,0.014662,0.023226,0.014662,0.003676,0.000363,0.001446,0.014662,0.058488,0.092651,
					0.058488,0.014662,0.001446,0.002291,0.023226,0.092651,0.146768,0.092651,0.023226,0.002291,
					0.001446,0.014662,0.058488,0.092651,0.058488,0.014662,0.001446,0.000363,0.003676,0.014662,
					0.023226,0.014662,0.003676,0.000363,0.000036,0.000363,0.001446,0.002291,0.001446,0.000363,
					0.000036 };
	Mat blurKernel = Mat(7, 7, DataType<double>::type, kernelVals);

	for (int row = 0; row < output.rows; row++) {
		for (int col = 0; col < output.cols; col++) {

			double LHS = calcEllipse(row, col, center, width, height,angle);

			//if at border blur
			if (LHS < 1.05 && LHS > 0.8) {
				Vec3b p = convolute(input, blurKernel, row, col);
				output.at<Vec3b>(row, col) = p;
			}

		}
	}
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
	Mat output = input.clone();

	double centerRow = cy;
	double centerCol = cx;

	for (int row = 0; row < input.rows; row++) {
		for (int col = 0; col < input.cols; col++) {
			double x = (col - centerCol) / s + centerCol;
			double y = (row - centerRow) / s + centerRow;

			if (x > 0 && x < input.cols && y > 0 && y < input.rows)
				output.at<Vec3b>(row, col) = input.at<Vec3b>(y, x);
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
			if (face.at<Vec3b>(row, col) != green) {
				//output.at<Vec3b>(row, col) = face.at<Vec3b>(row, col);
				int placeFaceHigher = input.rows / 20;
				if (row - placeFaceHigher >= 0)
					output.at<Vec3b>(row - placeFaceHigher, col) = face.at<Vec3b>(row, col);
			}
		}
	}

	cv::imwrite("Overlay.jpg", output);

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
Mat fetchFace(const Mat& input, int width, int height, Point center, double angle) {

	Mat onlyFace = input.clone();

	for (int row = 0; row < onlyFace.rows; row++) {
		for (int col = 0; col < onlyFace.cols; col++) {

			double LHS = calcEllipse(row, col, center, width, height,angle);

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
	double scaleFactor = 1.6;
	std::cout << "width: " << frame.cols << "height: " << frame.rows << endl;
	Point finalCenter;
	Mat faceOnly;
	Mat frame_gray;

	Mat overlayed = frame.clone();
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);


	//-- Detect faces
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces);

	for (auto val : faces)
		std::cout << val << " ";

	std::cout << "Number of faces detected : " << faces.size() << endl;

	// saves the center,width and height of each faces
	vector<pair<Point, pair<double, double>>> facePositions(faces.size());

	Point eye0, eye1;
	double angle = 0;
	Mat output = frame.clone();

	// For each face
	for (int i = 0; i < faces.size(); i++)
	{

		//Get eyes information
		Mat faceROI = frame_gray(faces[i]);
		vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes);
		std::cout << "No. of eyes: " << eyes.size() << endl;
		if (eyes.size() == 2)
		{
			eye0 = Point(eyes[0].x, eyes[0].y);
			eye1 = Point(eyes[1].x, eyes[1].y);
			angle = atan((double)(eye0.y - eye1.y) / (double)(eye0.x - eye1.x)) * 180 / PI;
			std::cout << "Angle is: " << angle << endl;
		}

		// Get face information
		int cx = faces[i].x + faces[i].width / 2;
		int cy = faces[i].y + faces[i].height / 2 - faces[i].height * 0.08;
		Point center(cx, cy);
		Point faceCenter(cx, cy);
		
		// Drawing ellipse
		Size faceSize(faces[i].width, (faces[i].height) * 1.3);
		RotatedRect rRect(faceCenter, faceSize, angle);
		ellipse(output, rRect, Scalar(0, 255, 0), 5);

		// An image with the cropped face with a green background
		faceOnly = fetchFace(frame, faces[i].width * 0.9, faces[i].height * 1.4, center, angle);

		// An image with the enlarged cropped face with a green background
		faceOnly = scale(faceOnly, cx, cy, scaleFactor);
		overlayed = overlay(faceOnly, overlayed);
	}

	// for loop to blur all edges of all faces
	output = overlayed.clone();
	for (int i = 0; i < faces.size(); i++) {
		output = blurCorner(output, facePositions[i].second.first, 
		facePositions[i].second.second, facePositions[i].first, angle);
	}

	displayImage(output, "blurred Corners");
	imwrite("afterBlur.jpg", output);
	return output;
}

// displayImage
//
// Pre-condition: img is a valid img
// Post-condition: Displays the image onto a popUp window
void displayImage(const Mat& img, String winName)
{
	namedWindow(winName, WINDOW_NORMAL);
	resizeWindow(winName, img.cols, img.rows);
	imshow(winName, img);
	waitKey(0);
}


int main(int argc, const char** argv) {
	cout << "runs" << endl;

	String filename = argv[1];
	cout << filename << endl;
	Mat original = imread(filename);
	while (original.rows * original.cols >= 500000) //more than 500 x 500
		resize(original, original, Size(), 0.5, 0.5);


	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(1)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(2)Error loading\n"); return -1; };
	//if (!smile_cascade.load(smile_cascade_name)) { printf("--(3)Error loading\n"); return -1; };

	//-- 2. Apply the classifier to the frame
	imwrite("original.jpg", original);
	Mat enlarged = detectAndDisplay(original); //this changed original fo some reason 
	imwrite("originalAfterEnlarged.jpg", original);


	//Mat overlayed = overlay(enlarged, original);
	//imwrite("originalAfterOverlay.jpg", original);
	cout << "Done";
	//imwrite("output.jpg", input);


	return 0;
}

