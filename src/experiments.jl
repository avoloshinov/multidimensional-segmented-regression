include("linear_merging_algorithm.jl")
include("synthetic_data_generation.jl")

function trial(k::Int, n::Int, d::Int, z::Int, sigma::Float64, upper_bound::Float64)
    levels = ceil(Int,log(2,n))
    y, ystar, X = generate_random_regression_data(k, n, d, z, sigma)
    leaves = fit_linear_merging(X, y, sigma, z, levels, k, upper_bound)
    yhat = leaves_to_yhat(X,leaves) #reconstruct the yhat from leaves
    println()
    println("number of samples: ", n)
    println("number of pieces in true function: ", k)
    println("number of pieces in output: ", length(leaves)) #this is the number of pieces (with samples) of output
    println("MSE: ", mse(yhat,ystar))
    println()
end

#trial(k::Int, n::Int, d::Int, z::Int, sigma::Float64, upper_bound::Float64)
println("Piecewise in 2 dimensions:")
println()
k=16

println("changes with increasing number of samples:")
@time trial(k, k*100, 5, 2, 1.0, convert(Float64,k))
@time trial(k, k*1000, 5, 2, 1.0, convert(Float64,k))

k=64
@time trial(k, k*100, 5, 2, 1.0, convert(Float64,k))
@time trial(k, k*200, 5, 2, 1.0, convert(Float64,k))

println()
println("changes with increasing number of pieces in output")
k=16
@time trial(k, k*100, 5, 2, 1.0, convert(Float64,k))
@time trial(k, k*1000, 5, 2, 1.0, convert(Float64,k/4))
@time trial(k, k*1000, 5, 2, 1.0, convert(Float64,k/8))


println("Piecewise in 3 dimensions")

k=64
@time trial(k, k*100, 5, 3, 1.0, convert(Float64,k))
@time trial(k, k*200, 5, 3, 1.0, convert(Float64,k/4))
