# Edge Detection 
This is software to allow the user to apply edge detection with a high pass gaussian filter with varying radii and standard deviations.

We first read in the image and then transform the image into the frequency domain. Then we shift the values in the frequency domain, such that they are centered around the center of the image.

Then we initialize a ideal low pass filter and apply a gaussian blurring filter to recieve a gaussian low pass filter. We then subtract this filter from 1 to receive a gaussian high pass filter. 

We then multiply the frequency spectrum of the image with the gaussian high pass filter. Then we shift the values back to their original position, and apply the inverse fast fourier transform to convert the image back to the time domain. 

The function finally returns the original image, as well as the high pass filtered image. 



## Using the Edge Detection Function
The edgeDetection( img, rad, sigma, roi, channel=None ) function takes in 5 arguments. 
- img: this is the image file. It can either be a .ORF raw file, a .jpg file or a numpy array.
- rad: here you specify the radius of the ideal low pass filter. This determines how much of the low frequency information is filtered out.
- sigma: here you specify the standard deviation of the gaussian blur filter that gets applied to the ideal low pass filter. 
- roi: specify the region of interest, that you would like to crop your image to. The region of interest is specified as an array roi = [x, y, w, h]. The first two value being the x and y values you would like to start at. The second two value being the widht and the hight of the region you would like to crop. 
- channel: you can specifiy which channel of the image you would like to apply the edge detection to. You can only apply the edge detection to single channel arrays. Therefore you have the option to apply the edge detection to one of the rgb channels of the image, by passing 'r', 'b' or 'g' as keyword arguments. Or you have the option apply the edge detection to a gray scale of the image by passing the argument 'gray'. 