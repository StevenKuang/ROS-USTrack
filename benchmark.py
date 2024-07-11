import timeit
import cv2
import numpy as np


# Dummy data for benchmarking
frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
pred_mask = np.random.randint(0, 256, (480, 640), dtype=np.uint8)

# Define the functions
def bitwise_and_method():
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.bitwise_and(grayscale_frame, grayscale_frame, mask=pred_mask)

def numpy_method():
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return grayscale_frame * (pred_mask // 255)

def multiply_method():
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.multiply(grayscale_frame, pred_mask // 255)

# Benchmark
print(timeit.timeit(bitwise_and_method, number=1000))
print(timeit.timeit(numpy_method, number=1000))
print(timeit.timeit(multiply_method, number=1000))
