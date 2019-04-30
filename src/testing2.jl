 #using PyPlot
 #using JLD

#include("dim_linear_merging.jl")

#function merging_k(X::Array{Float64,2}, y::Array{Float64,1}, k::Int)
#    return fit_linear_merging(X, y, sigma, k, floor(Int, k / 2.0), initial_merging_size=d)
#end

#sigma = 1.0
#n_vals = round(Int, exp10.(range(2.0, 4.0, 7)) #logspace(2.0, 4.0, 7))
#k = 5
#d = 10
#mses = Dict{ASCIIString, Array{Float64, 2}}()
#times = Dict{ASCIIString, Array{Float64, 2}}()
#n_warmup = 100
#y, ystar, X = generate_equal_size_random_regression_data(k, n_warmup, d, sigma);
#yhat_partition = merging_k(X, y, k)

#X=[1.0 2.0 3.0 10.0 ; 9.0 3.0 5.0 7.0; 8.0 2.0 8.0 6.8; 5.0 2.2 3.3 4.0]
#Y=[[8.0, 2.0, 8.0,6.8], [2.2,3.0,9.0,11.2],[1.0, 2.0, 3.0,7.0], [9.0, 3.0, 5.0,7.9]]
Y=[[8.0, 2.0, 8.8,6.8], [2.2,3.4,9.0,11.2],[1.0, 2.8, 3.0,7.0], [9.6, 3.5, 5.0,7.9],[13.6, 1.55, 5.22,7.99]]
X=[[8.0, 2.0, 8.8,6.8], [2.2,3.4,9.0,11.2],[1.0, 2.8, 3.0,7.0], [9.6, 3.5, 5.0,7.9],[13.6, 1.55, 5.22,7.99],[6.7,8.98,1.23,4.5],[4.55,6.44,1.11,2.22],[10.7,7.1,5.5,4.0],[4.2,3.4,6.21,9.43]]
Z=[[8.0, 2.0, 8.8,6.8], [2.2,3.4,9.0,11.2],[1.0, 2.8, 3.0,7.0], [9.6, 3.5, 5.0,7.9],[13.6, 1.55, 5.22,7.99],[6.7,8.98,1.23,4.5],[4.55,6.44,1.11,2.22]]
#Z=[[1.0, 2.8, 3.0,7.0], [9.6, 3.5, 5.0,7.9],[13.6, 1.55, 5.22,7.99],[6.7,8.98,1.23,4.5],[4.55,6.44,1.11,2.22],[10.7,7.1,5.5,4.0],[4.2,3.4,6.21,9.43]]
z=2
levels = 3

num_rectangles = 4
n = 7
d = 4
z = 2
sigma = 1.0

y, ystar, data = generate_random_regression_data(num_rectangles, n, d, z, sigma)

result = create_tree(data, y, z,levels)
print_tree(result,levels,false)
