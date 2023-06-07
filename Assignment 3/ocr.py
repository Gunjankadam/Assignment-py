import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Load the input image
image = cv2.imread('sample-image')  #name of image to be loaded

print("Shape of the image", image.shape)


#rotating image
angle = -10 # Angle in degrees for rotation
rows, cols, _ = image.shape
rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

# cropping image
crop = image[250:460, 490:1136]

#grayscaling image
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

#show preprocessed image
cv2.imshow('p-image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#apply tesseract
config = '--oem 3 --psm 6'
text1 = pytesseract.image_to_string(gray, config=config)

print(text1)