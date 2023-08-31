# Camera Imaging Pipeline
This is a python package to allow you to process raw sensor data from a camera into a visually appealing image. 

## 1. Selecting the parameters in the JSON file
- resize val: this parameter indicates the percentage by which you would like to increase or decrease the size of the image, for more comfortable viewing. The second return value (numpy array) of the applyToImage() method in the imageProcessing class will return the resized image. 
- exposure compensation: this parameter determines by how much the intensity values of the image will be rescaled. Values from 0.0. to 0.9 will decrease the brightness of the image. Values from 1.1 to 2.0 will increase the brightness of the image. 
- tone curve: the tone curve function currently incomplete. The idea was to create a look up table to be able to remap the intensity values to new values according to a specified curve.
- gamma correction: the gamma correction parameter determines which gamma curve is used to remap the intensity values. 
- denoising: the denoising parameter determines how much gaussian blurring occurs to smooth out the image. 
- processes: this is the section where you specify what processes you would like to apply to the image. Select 'true' if you would like to apply the process, and 'false' if you don't want the raw data to be effected by that processes. 

## 2. Running the processing pipeline
To apply the processing pipeline to raw image data, take the following steps. 
1. Specify the file path to the raw file.
2. Specify the file path to the .json file.
3. Open the .json file with the open() method and load the .json with the json.load() method. 
4. Initialize an image processing class by passing it the opened .json file.
5. Then pass the image path as a string to the applyToImage() method from the imageProcessing class. This methos returns two values: 
    - the non-resized image data 
    - and the resized image data for more comfortable viewwing wit the opencv imshow() function. 

Please refer to the example.py file to see an example of how to process a RAW image file. 