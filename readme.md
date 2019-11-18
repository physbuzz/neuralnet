Neural Network Project
===

The goal is to write some general neural network code. Currently the project is configured to classify the primes for `0<=n<1024`. See out.txt for example output.

This is just a first pass to make sure everything is behaving as expected and the error is actually minimized. Efficiency can be greatly improved.


TODO:
 1. Fix exploding gradients. Should be able overfit and get ~100% accuracy. So add relu/softmax/sigmoid.
 2. Improve performance using BLAS 
 3. Fix data importing to work with arbitrary datasets / arbitrary csvs.
