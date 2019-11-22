# BigHeadMode
performs BigHeadMode using OpenCV
(currently working on it...)

Detects all faces of people from a picture using OpenCV library and "saves" the region of interest as an ellipse.
In a new image, enlarges their faces using a scaling function and puts the enlarged head back into the original picture.
When putting the faces, uses bilinear interpolation and blends out the color so it looks natural.

