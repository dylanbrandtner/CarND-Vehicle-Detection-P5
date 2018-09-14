# Vehicle Detection Project

[//]: # (Image References)
[image1]: ./output_images/HOG_features.png  "HOG"
[image2]: ./output_images/Car_distance_examples.png  "Car Distances"
[image3]: ./output_images/Car_distance_windows.png  "Windows"
[image4]: ./output_images/found_cars.png  "Found Cars"
[image5]: ./output_images/heatmap.png  "Heat map"
[image6]: ./output_images/full_pipe.png "Full pipe"


The overall goal of this project was to overlay bounding boxes for vehicles onto a input video.  Here as an example frame from my final result:

![alt text][image6]

The specific goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a SVM classifier
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Source code

All source code referenced below is contained in the [IPython notebook](./Detection.ipynb) in this repo.  See that notebook to follow along with the code changes described here. 

## [Rubric Points](https://review.udacity.com/#!/rubrics/513/view)
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### Extracting HOG features from the training images

The code for this step is contained in the code cells of the IPython notebook under "Feature Extraction".  The 'get_hog_features()' and  'extract_hog_features()' functions were mostly taken from the course materials to extract a set of HOG features from a given image.      

In the "Read in Images" section, I started by reading in all the `vehicle` and `non-vehicle` images from the provided data set.  In total, there were 8792 Car images, and 8958 non-car images.  

I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.  Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]

#### Choosing HOG Parameters

I wanted to see how each of the color spaces and channels would fare when classifying the training data using a linear SVM. In the code section "Extract Feature Sets and Train a Linear SVM Classifier" I setup two functions to quickly extract all training and test sets using a given set of parameters, and then a function to train a linear classifer and report the training time and test accuracy.

In the "Find Optimal Color Space and Hog Channel(s)" section, I setup a wrapper function that would run the above mentioned functions on an input array of color spaces and channels.  Here are my results:

| Color Space  | HOG Channel | Accuracy  |
|:------------:|:-----------:|:---------:|
|     HLS      |     0       |  90.20%   |
|     HLS      |     1       |  95.41%   |
|     HLS      |     2       |  87.84%   |
|     HLS      |    ALL      |  98.06%   |
|     HSV      |     0       |  89.84%   |
|     HSV      |     1       |  88.32%   |
|     HSV      |     2       |  95.52%   |
|     HSV      |    ALL      |  98.34%   |
|     LUV      |     0       |  94.06%   |
|     LUV      |     1       |  92.06%   |
|     LUV      |     2       |  88.68%   |
|     LUV      |    ALL      |  98.03%   |
|     RGB      |     0       |  93.78%   |
|     RGB      |     1       |  95.61%   |
|     RGB      |     2       |  94.68%   |
|     RGB      |    ALL      |  96.93%   |
|    YCrCb     |     0       |  94.54%   |
|    YCrCb     |     1       |  91.50%   |
|    YCrCb     |     2       |  89.98%   |
|    YCrCb     |    ALL      |  98.23%   |
|     YUV      |     0       |  94.65%   |
|     YUV      |     1       |  91.69%   |
|     YUV      |     2       |  89.86%   |

Given this data, the examination of various color channels, and the training speeds, I chose YUV channel 0.  It had a high accuracy during the training, was a single channel (so that extracting the HOG features would be faster later), and it seemed to distinguish cars fairly well in my testing. 

#### Training the classifier

In the code section "Training the final classifier" I trained a SVM using the 'GridSearchCV()' function from scikitlearn module to find the best parameters.  This method chose a "RBF" kernel with a C value of 10, and reached a test accuracy of 99.44%.  

The 'GridSearchCV()' function took over 20 minutes to run, so at this point I stored my classifier in a pickle file that I could reload later. 

### Sliding Window Search

#### Implementing the sliding window search

I wanted to search different window positions at different scales.  To determine the sets of sliding windows, I first grabbed images from the project video with cars at different distances:

![alt text][image2]

Under the "Configure sliding window pipeline" code section, I setup a 'draw_boxes()' function (adapted from 'find_cars()' in lesson materials) to see how the sliding windows would ultimately look on the images.  I came up with the following window areas and scales:

|    Zone    | Y Search area  |  Window Scale  |
|:----------:|:--------------:|:--------------:|
|  Close     |    425, 650    |      3.5       |
| Mid-Close  |    400, 600    |       2        |
|  Mid-Far   |    400, 550    |      1.5       |
|   Far      |    400, 500    |       1        |

Note: Window scale is multiplied by a 64x64 window.

I plotted the resulting boxes on the image:

![alt text][image3]

I then adapted the 'find_cars()' function from the course materials to take an image, window parameters, a classifier, and HOG parameters and search for windows that give a positive result from the classifier.  I also setup a 'sliding_window_pipe()' function to run 'find_cars()' multiple times for each sliding window area found above.  I ran this function on "test5.jpg" from the test images and got the following result:

![alt text][image4]

#### Heat Maps and Rejecting False Positives

As you can see above, my classifier flagged a few false positives on the test image, but also correctly classified several sample windows on the real cars.  In the section "Heat Map", I setup a heat map from the resulting set of found windows by combining several functions from the course materials into a 'create_heat_map()' function.  In order to be part of the heat map, there must be enough overlapping windows to exceed the input threshold.  It then draws a final bounding box around "hot" areas.  I experimented with several thresholds, but with the number of overlapping detections on each car, I chose a value of 3.  This resulted in the following heat map and final image with bounding boxes drawn:

![alt text][image5]

## Video Pipeline

### Initial Pipeline

At this point, I was ready to combine the previous steps into a single image processing pipeline.  In the "Combined Initial Pipeline" section, I setup a 'process_image()' function to be used in moviepy's 'fl_image()' function, that applies a function to each image and replaces it with the result.  My pipeline detects all car windows using my 'sliding_window_pipe()', and then creates a heat map to draw a final set of bounding boxes on the found cars.  Also, for debugging, I added composite images of the heat map and the full set of detections from the classifier.  Finally, I added text to the screen to list how many cars are detected on screen.  I ran the pipeline on a frame from the test video and got this result: 

![alt text][image6]
---

#### Testing initial pipeline on project video

Here's a [link to my initial video result](./project_video_out.mp4).

### Improving the Pipeline

After viewing the initial results, it was clear that spurious detections were still exceeding my initial heat map threshold in some cases.  Rough patches in the road would sometimes trigger multiple overlapping detections which would show as a car.  My classifier could obviously use some more tuning to avoid these, but instead I chose to focus on separating them out based on the properties of cars on a highway. Since cars on a highway are traveling close to the same speed as the camera itself, they should stay in a more consistent location in each frame of the video image than a stationary object.

Thus, I setup an improved pipeline that stored a history of bounding boxes found in each frame.  I then pass these bounding boxes into the same heat map function discussed above, and I set the threshold based on the size of the history.  I tuned the history size and threshold and eventually settled on a history of 20 frames and a threshold of history_size-5 (ie. 15 at most parts of the image).  Thus, the final bounding box drawn would only be ones that overlapped in 75% of the previous 20 frames.     

#### Testing improved pipeline on project video

Here's a [link to my improved video result](./project_video_out_improved.mp4).  As you can see, the results are much improved.

---

## Discussion

In general, this exercise showed both the power and limitations of simple classifiers and traditional computer vision techniques.  I would be interested to see how much improvement I could get in initial detection by using a deep learning approach instead of a simple classifier.  I also found the heat mapping approach to be a great tool for filtering false positives when search areas overlap. 

### Areas for Improvement

#### Efficiency 
I was surprised by how slow my pipeline was.  I initially chose an "ALL" HOG channel and got more consistent results when applying my classifier to test images, but the training, HOG extraction, and prediction on all 3 channels was so slow it would have taken _several_ hours to process the full project video (even after training was complete).  This kind of turnaround time was not conducive to any experimentation, and made me feel that my approach would not be usable in a real world environment.  Thus, I switched back to a single HOG channel.  Perhaps a GPU could complete these operations fast enough to be able to process a video stream in real time, but it seemed like a stretch if all channels were used.  I also thought that a less exhaustive sliding window set could be used to reduce the amount of images that needed to be processed, but when experimenting with this, I struggled to cleanly separate the cars with the heat map threshold due to the relatively high rate of false positives.   

#### Detection accuracy
Although I had an extremely high test accuracy on the training data, I also notice a fairly high instance of false positives when applied to the video.  Thus, I assume my classifier was not generalizing well.  I could have tried additional HOG parameter tuning, SVM parameter tuning, or additional features (spatial, histogram, etc) to improve this.  Also, as noted in the tips for the project, the input training data included some time series data, which may have lead to overfitting, depending on the random split of training/test data.  I could have manually separate the data in a better way and/or include more labeled images from other sources.  

