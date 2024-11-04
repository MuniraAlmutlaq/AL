import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd= r'D:/TesseractOCR/tesseract.exe'
import re
import os

TEMP_DATABASE= 'Website/TemporaryDatabase'

#Reading the uploaded image.
image_path= os.path.join(TEMP_DATABASE, 'extracted_image.png')
image= cv2.imread(image_path)

#Preprocessing the image for OCR and resizing the image to improve OCR accuracy.
image= cv2.resize(image, None, fx= 2, fy= 2, interpolation= cv2.INTER_LINEAR)

#Converting the image to grayscale.
gray_image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Increasing the contrast by adjusting gamma.
gamma= 1.5
adjusted= np.array(255 * (gray_image / 255) ** gamma, dtype= 'uint8')

#Applying Gaussian blur to reduce noise.
blurred_image= cv2.GaussianBlur(adjusted, (5, 5), 0)

#Applying Otsu's thresholding to binarize the image.
_, thresh_image= cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#Applying dilation to enhance the text structure.
kernel= np.ones((2, 2), np.uint8)
dilated_image= cv2.dilate(thresh_image, kernel, iterations= 1)

#Extract the text from the processed image.
text= pytesseract.image_to_string(dilated_image, lang= 'ara')

#Removing special characters (keep common punctuation and Arabic letters).
text= re.sub(r"[^\w\s,.?!\u0600-\u06FF]", "", text)  #Keeps Arabic characters.

#Preprocessing the extracted text and removing extraneous line breaks.
text= text.replace("\n", " ")  #Replacing line breaks with spaces.
text= text.replace("\r", " ")  #Handling carriage returns.
text= " ".join(text.split())   #Ensuring consistent spacing.

#Normalize whitespace.
text= " ".join(text.split())  #Removes extra spaces.

#Writing the processed image in arabic_text.txt document.
arabic_text_file= os.path.join(TEMP_DATABASE, 'arabic_text.txt')
with open(arabic_text_file, 'w', encoding= 'utf-8') as file:
    file.write(text)
    print("The preprocessed arabic text has already written in arabic_text.txt")
