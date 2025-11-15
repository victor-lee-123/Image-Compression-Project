import cv2

img = cv2.imread("images/test.jpg")

if img is None:
    print("Failed to load image!")
else:
    print("Image loaded successfully:", img.shape)
    cv2.imshow("Test Image", img)
    cv2.waitKey(0)
