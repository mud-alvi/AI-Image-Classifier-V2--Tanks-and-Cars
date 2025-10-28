# A Basic image classifier I made as a way to distinguish between tanks and cars

- This whole classifier was coded on jupyter because I found this out that it was way easier to code in modules esp for training and checking accuracy

 # TO VIEW THE CLASSIFIER ITSELF
 - just click on the website and it will take you to huggingface where I have already done the work. :)

  # TO CREATE THE CLASSIFIER YOUR SELF
  1. firstly create a folder called "Z_Image" on your desktop.
  2. add a "train" folder and a "cars_tanks" excel sheet which can be taken from above.
  3. then take the main source code cited above and execute it as a whole on VS code and at the end there is an option to save it, do that.
  4. now you will have your own pth file within the "Z_Image" folder.
  OR IF YOU DONT WANT TO DO THE ABOVE JUST INSTALL THE READY PTH FILE FROM ABOVE
  6. now open huggingface on your browser and create a space. Furthermore, in the files section of the space, add the pth file, app.py file and requirements.txt file which can be copied from above
  7. and then let the huggingface application do the work by creating the website
  8. It will take maximum of approx 10 mins and you wiil be met by the interface where you can upload an image of your liking and the classifier will tell you if it looks like a tank or a car :O
- so yea enjoy :)


# Explaination about the versions 
1. V1 :- it was the prototype version with a total of 3.2k images and an accuracy of 49%
2. V2 :- it was the actual final pre released product featuring 6k images and an accuracy of 90.25% with more epochs and a better dataset with equal amount of images of tanks and of cars.
3. V2.1 :- latest version (made only because i lost my orignal dataset) so I increased the dataset to the total of 12k images and also added a validation set increasing my accuracy up to 91.28%
