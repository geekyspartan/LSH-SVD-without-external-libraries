# LSH and SVD without using External Libraries

Implemented Dimensionality Reduction using Singular Vector Decomposition and Locality Sensitive Hashing to find similar files

Run code using pyspark and pass directory path as an argument.This code finds the similarity between images using following steps:
1. Read data (tif files) and convert them into arrays
2. Convert the 4 values (r,g,b,intensity) to single value using
intensity = int(rgb_mean * (infrared/100))
3. Divide the image into 500x500 subimages. We will find similary between these subimages.
4. Take mean intensity of (10x10) sub-matrices and convert 500*500 to 50*50.
5. Computer row difference and column difference of intensities
6. Convert all values < -1 to -1, those > 1 to 1, and the rest to 0 and flatten the values to get array of 1*4900 size.
7. Create a 128 length signature for each feature vector
8. Apply LSH and find sub-images similarity.


Test by tweeking number of bands, signature length, hash function. Checkout [Sample Output](https://github.com/techiepanda/SVD-LSH-no-libraries/blob/master/output.txt)
