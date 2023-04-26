# k_nearest_neighbors
Implementation of k nearest neighbor query using kd-trees. In 2-dimensional data it runs in O(logn) time

pip install matplotlib and datetime fore running. 
program also uses standard python3 library packages: time, typing, random, math, datetime, csv.

To run, download file and run with "python3 knn.py" terminal command. 
Program will prompt how many tests to run, each test becomung progressively larger input size. 
For n tests, the largest input size will be a set of 2^(n-1) points.

Program generates random set of points for each of the n tests, performs a nearest neighbor query. 
For input size n, the program runs 100 tests and and prints input size, average number of tree nodes visited in query, 
and average total runtime of query in microseconds for proof of O(logn) runtime.

After running tests, it will prompt user preference to see a visualization of queery results. 
User can type "Y" or "y" and hit enter. Enter test size as an integer and hit enter (visually best if under 400). 
Blue is target and red is nearest neighbor, all other points are black.

Close popup matplot window or Ctrl C to end run.
