**Vehicle Detection Project**

[//]: # (Image References)
[image1]: ./examples/HOG_features.png

The overall goal of this project was to overlay bounding boxes for vehicles onto a input video.  Here as an example frame from my final result:

The specific goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a SVM classifier
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Source code

All source code referenced below is contained in the [IPython notebook](./Detection.ipynb) in this repo.  See that notebook to follow along with the code changes described here. 

## [Rubric Points](https://review.udacity.com/#!/rubrics/513/view)
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### Extracting HOG features from the training images

The code for this step is contained in the code cell of the IPython notebook under "Feature Extraction".  The get_hog_features() and  extract_hog_features() functions were mostly taken from the course materials to extract a set of HOG features from a given image.      

In the "Read in Images" section, I started by reading in all the `vehicle` and `non-vehicle` images from the provided data set.  In total, there were 8792 Car images, and 8958 non-car images.  

I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.  Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]

#### HOG paramater choice

I wanted to see how each of the color spaces and channels would fare when classifying the training data using a linear SVM. In the code section "Extract Feature Sets and Train a Linear SVM Classifier" I setup two functions to quickly extract all training and test sets using a given set of parameters, and then a function to train a linear classifer and report the training time and test accuracy.

In the "Find Optimal Color Space and Hog Channel(s)" section, I setup a wrapper function that would run the above mentioned functions on an input array of color spaces and channels.  Here are my results:

Color Space: HOG Channel | Accuracy
========================================
     HLS:0               | 90.20%
     HLS:1               | 95.41%
     HLS:2               | 87.84%
     HLS:ALL             | 98.06%
     HSV:0               | 89.84%
     HSV:1               | 88.32%
     HSV:2               | 95.52%
     HSV:ALL             | 98.34%
     LUV:0               | 94.06%
     LUV:1               | 92.06%
     LUV:2               | 88.68%
     LUV:ALL             | 98.03%
     RGB:0               | 93.78%
     RGB:1               | 95.61%
     RGB:2               | 94.68%
     RGB:ALL             | 96.93%
     YCrCb:0             | 94.54%
     YCrCb:1             | 91.50%
     YCrCb:2             | 89.98%
     YCrCb:ALL           | 98.23%
     YUV:0               | 94.65%
     YUV:1               | 91.69%
     YUV:2               | 89.86%
     YUV:ALL             | 98.00%
========================================

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

