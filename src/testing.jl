include("linear_merging_algorithm.jl")
include("synthetic_data_generation.jl")

#X=[1.0 2.0 3.0 10.0 ; 9.0 3.0 5.0 7.0; 8.0 2.0 8.0 6.8; 5.0 2.2 3.3 4.0]
#Y=[[8.0, 2.0, 8.0,6.8], [2.2,3.0,9.0,11.2],[1.0, 2.0, 3.0,7.0], [9.0, 3.0, 5.0,7.9]]
#Y=[[8.0, 2.0, 8.8,6.8], [2.2,3.4,9.0,11.2],[1.0, 2.8, 3.0,7.0], [9.6, 3.5, 5.0,7.9],[13.6, 1.55, 5.22,7.99]]
#X=[[8.0, 2.0, 8.8,6.8], [2.2,3.4,9.0,11.2],[1.0, 2.8, 3.0,7.0], [9.6, 3.5, 5.0,7.9],[13.6, 1.55, 5.22,7.99],[6.7,8.98,1.23,4.5],[4.55,6.44,1.11,2.22],[10.7,7.1,5.5,4.0],[4.2,3.4,6.21,9.43]]
#Z=[[8.0, 2.0, 8.8,6.8], [2.2,3.4,9.0,11.2],[1.0, 2.8, 3.0,7.0], [9.6, 3.5, 5.0,7.9],[13.6, 1.55, 5.22,7.99],[6.7,8.98,1.23,4.5],[4.55,6.44,1.11,2.22]]
#Z=[[1.0, 2.8, 3.0,7.0], [9.6, 3.5, 5.0,7.9],[13.6, 1.55, 5.22,7.99],[6.7,8.98,1.23,4.5],[4.55,6.44,1.11,2.22],[10.7,7.1,5.5,4.0],[4.2,3.4,6.21,9.43]]

k = 16       # true number of pieces/rectangles of function
n = k*600    # number of samples
d = 5         # dimension of samples
z = 2         # the number of dimensions the piecewise functions are defined in
sigma = 1.0
levels = ceil(Int,log(2,n))
upper_bound = convert(Float64,k) # should be 2*k*(log(2,n)^2) in theory
#
# y, ystar, X = generate_random_regression_data(k, n, d, z, sigma)
#
# #root, leaves = create_tree(data, y, z,levels)
# #print_tree(root,levels,true)
#
# leaves = fit_linear_merging(X, y, sigma, z, levels, k, upper_bound)
# yhat = leaves_to_yhat(X,leaves) #reconstruct the yhat from leaves
#
# println(length(leaves)) #this is the number of pieces (with samples) of output
# mse(yhat,ystar)


function trial(k::Int, n::Int, d::Int, z::Int, sigma::Float64, upper_bound::Float64)
    levels = ceil(Int,log(2,n))
    y, ystar, X = generate_random_regression_data(k, n, d, z, sigma)
    leaves = fit_linear_merging(X, y, sigma, z, levels, k, upper_bound)
    yhat = leaves_to_yhat(X,leaves) #reconstruct the yhat from leaves
    # println()
    # println("number of samples: ", n)
    # println("number of pieces in true function: ", k)
    # println("number of pieces in output: ", length(leaves)) #this is the number of pieces (with samples) of output
    # println("MSE: ", mse(yhat,ystar))
    # println()
    return mse(yhat,ystar)
end

#trial(k::Int, n::Int, d::Int, z::Int, sigma::Float64, upper_bound::Float64)
# println("Piecewise in 2 dimensions:")
# println()
# k=16
#
# println("changes with increasing number of samples:")
# @time trial(k, k*100, 5, 2, 1.0, convert(Float64,k))
# @time trial(k, k*1000, 5, 2, 1.0, convert(Float64,k))
#
# k=64
# @time trial(k, k*100, 5, 2, 1.0, convert(Float64,k))
# @time trial(k, k*200, 5, 2, 1.0, convert(Float64,k))
#
# println()
# println("changes with increasing number of pieces in output")
# k=16
# @time trial(k, k*100, 5, 2, 1.0, convert(Float64,k))
# @time trial(k, k*1000, 5, 2, 1.0, convert(Float64,k/4))
# @time trial(k, k*1000, 5, 2, 1.0, convert(Float64,k/8))
#
#
# println("Piecewise in 3 dimensions")
#
# k=64
# @time trial(k, k*100, 5, 3, 1.0, convert(Float64,k))
# @time trial(k, k*200, 5, 3, 1.0, convert(Float64,k/4))
