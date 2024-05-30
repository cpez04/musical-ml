import cv2
import numpy as np

# Transforms the image to a binary (GRAYSCALE) image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return binary_image

# Removes the staff lines from the image
def remove_staff_lines(image):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    return cv2.subtract(image, detected_lines)

def segment_notes(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    notes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        note = image[y:y+h, x:x+w]
        notes.append(note)
    return notes




if __name__ == '__main__':
    image_path = 'test.png'
    binary_image = preprocess_image(image_path)
    
    removed_staff = remove_staff_lines(binary_image)
    
    # Display the images
    cv2.imshow('Original Image', binary_image)
    cv2.imshow('Removed Staff Lines', removed_staff)


    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  