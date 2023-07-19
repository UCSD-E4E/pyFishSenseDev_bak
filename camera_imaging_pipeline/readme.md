# Camera Imaging Pipeline
This repository contains a pyhton implementation of the camera imaging pipeline to process raw sensor data into a viewable image. The code is based on the paper corresponding to the following GitHub repository: file:///C:/Users/Hamish/Zotero/storage/Q9C8Z22E/camera-pipeline.html.


## Description - High Level
An instance of the imageProcessing class is obtained by passing it a dictionaty with the desired processing parameters during initiation. Then the file path to a raw file (in our case .ORF files) is passed to the  applyToImage() method, which returns two values, the processed raw data in it's original shape, and a resized image for comfortable viewing of the processed image. Both return values are numpy arrays with 8-Bit integer elements.  
The main processing functions are found in utils/processing_functions.py. Helper functions are found in utils/helpers.py and utils/analysis.py. The main class structure is found in src/image_processing.py.

## Steps to use
1. Edit (or create if not already available) the params.json file to select the steps of the pipeline you would like to apply to the raw sensor data, and adjust the parameters of each step to fit your needs. 
    - 'resize_val': resize the output image to a comfortable viewing size
    - 'exposure_compenastion': change the overall brightness of the image by scaling the intensity values. Values in the range from 0-1 will darken the     image, and values between 1 and 2 will brighten the image. 
    - 'tone_cruve': is not currently in use
    - 'gamma_correction': remap the initial intensity values according to the gamma function: https://docs.opencv.org/3.4/Basic_Linear_Transform_Tutorial_gamma.png
    - 'denoising': adjust the denoising factor which adjusts the intensity of blurring occuring due to convoling a low pass filter with the image data
    - 'colour': select whether or not you would like to have a greyscale image or an RGB image
    - 'processes': select which steps of the image processing pipeline you would like to apply to your sensor data 
2. Unpack the params.json file with the json module.
3. Pass the resulting dictionary during class initiation to imageProcessing().
4. Pass the file path of the raw file you would like to process to the applyToImage() method. This returns two values, the first one being the processed sensor data in it's original shape. The second one being the reshaped image for comfortable viewing. 
(5. To display the image, pass the return value from the applyToImage() method to cv2.showImage())

You can find an example of how to use the imageProcessing class in cip_example.py.