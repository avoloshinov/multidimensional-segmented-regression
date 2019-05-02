#X=[1.0 2.0 3.0 10.0 ; 9.0 3.0 5.0 7.0; 8.0 2.0 8.0 6.8; 5.0 2.2 3.3 4.0]
#Y=[[8.0, 2.0, 8.0,6.8], [2.2,3.0,9.0,11.2],[1.0, 2.0, 3.0,7.0], [9.0, 3.0, 5.0,7.9]]
#Y=[[8.0, 2.0, 8.8,6.8], [2.2,3.4,9.0,11.2],[1.0, 2.8, 3.0,7.0], [9.6, 3.5, 5.0,7.9],[13.6, 1.55, 5.22,7.99]]
#X=[[8.0, 2.0, 8.8,6.8], [2.2,3.4,9.0,11.2],[1.0, 2.8, 3.0,7.0], [9.6, 3.5, 5.0,7.9],[13.6, 1.55, 5.22,7.99],[6.7,8.98,1.23,4.5],[4.55,6.44,1.11,2.22],[10.7,7.1,5.5,4.0],[4.2,3.4,6.21,9.43]]
#Z=[[8.0, 2.0, 8.8,6.8], [2.2,3.4,9.0,11.2],[1.0, 2.8, 3.0,7.0], [9.6, 3.5, 5.0,7.9],[13.6, 1.55, 5.22,7.99],[6.7,8.98,1.23,4.5],[4.55,6.44,1.11,2.22]]
#Z=[[1.0, 2.8, 3.0,7.0], [9.6, 3.5, 5.0,7.9],[13.6, 1.55, 5.22,7.99],[6.7,8.98,1.23,4.5],[4.55,6.44,1.11,2.22],[10.7,7.1,5.5,4.0],[4.2,3.4,6.21,9.43]]

k = 4         # true number of pieces/rectangles of function
n = 10000     # number of samples
d = 4         # dimension of samples
z = 2         # the number of dimensions the piecewise functions are defined in
              # todo something breaks when z=3 right now....
sigma = 1.0
levels = ceil(Int,log(2,n))

y, ystar, X = generate_random_regression_data(k, n, d, z, sigma)

#root, leaves = create_tree(data, y, z,levels)
#print_tree(root,levels,true)

leaves = fit_linear_merging(X, y, sigma, z, levels, k)
length(leaves)
