# fruit_condition_classifier
FRUTSH - A prototype built with the help of Nvidia Jetson and Raspberry Pi Cam V2 that classifies fruits as fresh or rotten with the help of a CNN model and displays the prediction on a web app and an android app.

**ESS LABORATORY REPORT**

*Fruit Classification & Deterioration Check*

**OBJECTIVE:** To identify and classify rotten fruit from good ones with
the aim of reducing food wastage.

**TECHNOLOGY USED:**

**Hardware :**

*Nvidia Jetson*

-   NVIDIA Jetson systems provide the performance and power efficiency
    to run autonomous machines software, faster and with less power.
    Each is a complete System-on-Module (SOM), with CPU, GPU, PMIC,
    DRAM, and flash storage—saving development time

-   NVIDIA Jetson Nano Developer Kit is a small, powerful computer that
    lets you run multiple neural networks in parallel for applications
    like image classification, object detection, segmentation, and
    speech processing.

-   All in an easy-to-use platform that runs in as little as 5 watts

-   Jetson is also extensible.

    ![](media/image1.jpeg){width="2.7576541994750654in"
    height="2.249785651793526in"}

*Camera Module – Raspberry Pi Cam V2*

![](media/image2.png){width="2.00833552055993in"
height="1.9928565179352582in"}

-   Consists of a Sony IMX219 8-megapixel sensor.

-   Can be used to take high-definition video, as well as stills
    photographs, time-lapse, slow-motion videos etc.

-   Option to create effects using additional libraries.

-   More than just a resolution upgrade: it’s a leap forward in image
    quality, color fidelity, and low – light performance

-   Supports 1080p30, 720p60 and VGA90 video modes, as well as still
    capture and attaches via a 15cm ribbon cable to the CSI port

> ![](media/image3.png){width="3.5353193350831145in"
> height="2.236111111111111in"}

**Frameworks and libraries:**

*Tensorflow*

-   TensorFlow is an end-to-end open source platform for
    machine learning. It has a comprehensive, flexible ecosystem of
    tools, libraries and community resources that lets researchers push
    the state-of-the-art in ML and developers easily build and deploy ML
    powered applications

-   TensorFlow offers multiple levels of abstraction so you can choose
    the right one for your needs. Build and train models by using the
    high-level Keras API, which makes getting started with TensorFlow
    and machine learning easy.

-   If you need more flexibility, eager execution allows for immediate
    iteration and intuitive debugging. For large ML training tasks, use
    the Distribution Strategy API for distributed training on different
    hardware configurations without changing the model definition.

-   TensorFlow also supports an ecosystem of powerful add-on libraries
    and models to experiment with

*Numpy:*

-   Fast and versatile, the NumPy vectorization, indexing, and
    broadcasting concepts are the de-facto standards of array
    computing today.

-   NumPy offers comprehensive mathematical functions, random number
    generators, linear algebra routines, Fourier transforms, and more.

-   NumPy supports a wide range of hardware and computing platforms, and
    plays well with distributed, GPU, and sparse array libraries.

-   The core of NumPy is well-optimized C code. Enjoy the flexibility of
    Python with the speed of compiled code.

*REQUESTS MODULE:*

-   Requests allows you to send HTTP/1.1 requests extremely easily.
    There’s no need to manually add query strings to your URLs, or to
    form-encode your PUT & POST  data — but nowadays, just use the
    JSON method.

-   Requests is ready for the demands of building robust and reliable
    HTTP–speaking applications, for the needs of today.

-   Keep-Alive & Connection Pooling

-   International Domains and URLs

-   Sessions with Cookie Persistence

-   Browser-style TLS/SSL Verification

-   Basic & Digest Authentication

-   Familiar dict–like Cookies

-   Automatic Content Decompression and Decoding

-   Multi-part File Uploads

-   SOCKS Proxy Support

-   Connection Timeouts

-   Streaming Downloads

-   Chunked HTTP Requests

*Client side – Dedicated Web App & Native Android App.*

Web app using DJANGO:

Django is a high-level Python Web framework that encourages rapid
development and clean, pragmatic design. Built by experienced
developers, it takes care of much of the hassle of Web development, so
you can focus on writing your app without needing to reinvent the wheel.
It’s free and open source.

![](media/image4.png){width="5.360200131233595in"
height="2.9166666666666665in"}

*Android app Using android studio:*

![](media/image5.png){width="1.4027777777777777in"
height="2.8834886264216975in"}![](media/image6.png){width="1.3918919510061243in"
height="2.861111111111111in"}![](media/image7.png){width="1.3888888888888888in"
height="2.854938757655293in"}

**METHODOLOGY:**

The first step before training our model is data preprocessing. Over
here we take our dataset and then split into 2 datasets, Training and
Test set. We convert all our images into Grayscale and apply data
augmentation to increase the size of our network.

***Training the CNN model:***

A CNN is made up of multiple layers of neurons, each of which is a
nonlinear operation on a linear transformation of the preceding layer's
outputs. The layers mainly include convolutional layers and pooling
layers. The convolutional layers have weights that need to be trained,
while the pooling layers transform the activation using a fixed
function.In the CNNs, alternating convolutional layers, pooling layers,
full connection layers, and a final classification layers are
included.Convolutional features are represented through computing the
inner product of original fingerprint image and filters, and the process
of convolution is considered as the process of feature extraction.

1\) First convolution layer

A convolutional layer contains a set of filters whose parameters need to
be learned. The height and weight of the filters are smaller than those
of the input volume. Each filter is convolved with the input volume to
compute an activation map made of neurons. In other words, the filter is
slid across the width and height of the input and the dot products
between the input and filter are computed at every spatial
position.The convolutional layer computes the convolutional operation of
the input images using kernel filters to extract fundamental features.
The kernel filters are of the same dimension but with smaller constant
parameters as compared to the input images. The output volume of the
convolutional layer is obtained by stacking the activation maps of all
filters along the depth dimension.Convolution layer is responsible for
extracting features from our images by using a filter.We have used a 3x3
filter with 32 filters and a input image size of 64x64.The
multiplication of the filter with the image gives an output matrix which
we call feature map.

2\) Pooling layer

A pooling layer is usually incorporated between two
successive convolutional layers. It has been observed that the feature
map has a lot of parameters which can result in overfitting of our
model.The pooling layer reduces the number of parameters and computation
by down-sampling the representation. The pooling function can be max or
average. Max pooling is commonly used as it works betterAfter the
convolution operation, The Pooling used here is Max Pooling.Max Pooling
will find the largest value and store that and discard the remaining
values.This will keep only the most important parameters.  In classical
CNN, convolution layers are followed by a subsampling layer. The size of
effective maps is reduced by pooling layer, and some invariance features
are introduced.\]. The output of a max-pooling layer is given by the
maximum activation over non-overlapping regions, instead of averaging
the inputs as in a classical subsampling layer. A bias is added to the
resulting pooling and the output map is passed through the squashing
function.

3\) Flatten layer

Flattening is converting the data into a 1-dimensional array for
inputting it to the next layer. We flatten the output of the
convolutional layers to create a single long feature vector. And it is
connected to the final classification model, which is called
a fully-connected layer. In other words, we put all the pixel data in
one line and make connections with the final layer.Flatten is the
function that converts the pooled feature map to a single column that is
passed to the fully connected layer.

4\) Fully connected layer

Fully Connected Layer is simply, [feed forward *neural
networks*](https://en.wikipedia.org/wiki/Feedforward_neural_network)*. *Fully
Connected Layers form the last few layers in the network.The input to
the fully connected layer is the output from the *final* Pooling or
Convolutional Layer, which is flattened and then fed into the fully
connected layer.Input neuron layer is 128, hidden layer is 256 and
output neuron is 6 because we are training our model for 6 classes. We
compile our model and then start training it for 30 epochs.After
training, the weights are saved for prediction.

**STEPS:**

1.  To use our saved model again, we will load the weights by using the
    load\_model function from keras.

2.  The image is captured by using the camera attached to the NVIDIA
    JETSON NANO. All the recently captured images are renamed to a
    standard name so that we don't have to manually rename the
    input images.

3.  We take the input image, resize it to 64x64,and then pass it to our
    model for prediction.

4.  Over here will be a array label for predicting with size 6
    elements - \[0,0,0,0,0,0\].

5.  The result is given such that if result \[0\]\[1\] ==1 means fresh
    banana, it means that at the first index location of the
    array(\[0,1,0,0,0,0\])and it means that an fresh banana has
    been predicted.

6.  We can find the location index for our classes by using the
    class\_indices function.

7.  The predicted result is uploaded with the Requests library
    from Python.

8.  The request library pulls the image and the prediction and gives it
    to the web app for displaying.

9.  In this webapp, we have created a link where prediction and images
    captured by camera module will be uploaded on the server and then
    stored in the database.At the time of uploading, time is also
    noted.Link for uploading prediction is
    http://www.frutsh.pythonanywhere.com/postData.CNN model uses request
    module to upload the data to the webapp.

10. Request module pulls out the fruit name, condition and image from
    the CNN model’s output and then uploads on the webapp using
    post request.

11. When the data is uploaded on the webapp, then recent predictions of
    the fruits are fetched from the database and displayed on the
    homepage of our webapp.

12. The app downloads the html content of the webpage and extracts the
    useful data such as name, condition, image URL and time when the
    image of the fruit was captured.

13. Image URL is used to download the images.

14. Whenever the changes takes place on webapp, the same changes is
    reflected on the android app.

![](media/image8.jpeg){width="2.0645461504811897in" height="1.78125in"}
![](media/image9.png){width="1.666554024496938in"
height="3.3020833333333335in"}

Changes on webapp are reflected on android app

*CONTRIBUTION BY GROUP MEMBERS(GROUP 9):*

Nehanshu Tripathi – Hardware interfacing, implementation of the CNN
model and development of a seamless script running on the development
board (Nvidia Jetson).

Rahul Singh – Physical connections on the hardware, CNN model, Camera
interfacing with the development board as well as script to capture
image and then access it inside the model.

Niraj Dhotre – Implementation using the requests module to pull the
input image and the prediction from the model and upload it on the web
app.

Ajit Rakate – Development of a personalised web app which pulls info
from the requests module and displays the input image, prediction and
the time it was captured.

Aditya Singh – Development of a dedicated android app that pulls info
from the web app and displays the image, prediction and time when it was
captured. Implementation of a notification system to notify the users
about the condition of the fruits inside the fridge.

**GITHUB LINK**:
https://github.com/Nehanshu24/fruit\_condition\_classifier
