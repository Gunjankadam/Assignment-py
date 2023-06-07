import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load the input image
image = cv2.imread('input-image') #name of image to be loaded

#define functions
def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def binarize(image, threshold):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def remove_noise(image, kernel_size):
    denoised = cv2.medianBlur(image, kernel_size)
    return denoised

def dilate(image, kernel_size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilated = cv2.dilate(image, kernel, iterations=iterations)
    return dilated

def erode(image, kernel_size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    eroded = cv2.erode(image, kernel, iterations=iterations)
    return eroded

# Check if image needs to be cropped
should_crop = False # Set this flag to True if cropping is needed
if should_crop:
    # Shape of the image
    print("Shape of the image", image.shape)

    # [rows, columns]
    crop = image[110:500, 20:490]
image = crop

# Check if image needs to be rotated
should_rotate = False  # Set this flag to True if rotation is needed
if should_rotate:
    angle = 180  # Angle in degrees for rotation
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

# Grayscale conversion
gray_image = grayscale(image)

# Binarization
threshold_value = 127 
binary_image = binarize(gray_image, threshold_value)

# Noise removal
kernel_size_value = 3  
denoised_image = remove_noise(binary_image, kernel_size_value)

# Dilation
dilation_kernel_size = (3, 3) 
dilation_iterations = 1  
dilated_image = dilate(denoised_image, dilation_kernel_size, dilation_iterations)

# Erosion
erosion_kernel_size = (3, 3) 
erosion_iterations = 1 
eroded_image = erode(dilated_image, erosion_kernel_size, erosion_iterations)

# Save the output images
cv2.imwrite('grayscale_image.jpg', gray_image)
cv2.imwrite('binarized_image.jpg', binary_image)
cv2.imwrite('denoised_image.jpg', denoised_image)
cv2.imwrite('dilated_image.jpg', dilated_image)
cv2.imwrite('eroded_image.jpg', eroded_image)

# Display the results
cv2.imshow('Original Image', crop)
cv2.imshow('Grayscale Image', gray_image)
cv2.imshow('Binarized Image', binary_image)
cv2.imshow('denoised Image', denoised_image)
cv2.imshow('Dilated Image', dilated_image)
cv2.imshow('Eroded Image', eroded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#apply tesseract
config = '--oem 3 --psm 6'
text1 = pytesseract.image_to_string(image, config=config)
text2 = pytesseract.image_to_string(gray_image, config=config)
text3 = pytesseract.image_to_string(binary_image, config=config)
text4 = pytesseract.image_to_string(denoised_image, config=config)
text5 = pytesseract.image_to_string(dilated_image, config=config)
text6 = pytesseract.image_to_string(eroded_image, config=config)
print("--------------------")
print(text1)
print("--------------------")
print(text2)
print("--------------------")
print(text3)
print("--------------------")
print(text4)
print("--------------------")
print(text5)
print("--------------------")
print(text6)
print("--------------------")