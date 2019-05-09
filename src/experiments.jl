include("linear_merging_algorithm.jl")
include("synthetic_data_generation.jl")
using Statistics

function merging_k(X::Array{Array{Float64,1},1}, y::Array{Float64,1}, k::Int, n::Int)
    levels = ceil(Int,log(2,n))
    leaves = fit_linear_merging(X, y, sigma, z, levels, k, convert(Float64,k))
    yhat = leaves_to_yhat(X,leaves) #reconstruct the yhat from leaves
    return yhat, length(leaves)
end

function merging_kover2(X::Array{Array{Float64,1},1}, y::Array{Float64,1}, k::Int, n::Int)
    levels = ceil(Int,log(2,n))
    leaves = fit_linear_merging(X, y, sigma, z, levels, k, convert(Float64,k/2))
    yhat = leaves_to_yhat(X,leaves) #reconstruct the yhat from leaves
    return yhat, length(leaves)
end

function merging_kover4(X::Array{Array{Float64,1},1}, y::Array{Float64,1}, k::Int, n::Int)
    levels = ceil(Int,log(2,n))
    leaves = fit_linear_merging(X, y, sigma, z, levels, k, convert(Float64,k/4))
    yhat = leaves_to_yhat(X,leaves) #reconstruct the yhat from leaves
    return yhat, length(leaves)
end

function merging_kover8(X::Array{Array{Float64,1},1}, y::Array{Float64,1}, k::Int, n::Int)
    levels = ceil(Int,log(2,n))
    leaves = fit_linear_merging(X, y, sigma, z, levels, k, convert(Float64,k/8))
    yhat = leaves_to_yhat(X,leaves) #reconstruct the yhat from leaves
    return yhat, length(leaves)
end

algos = Dict([("merging_k", merging_k), ("merging_kover2", merging_kover2), ("merging_kover4", merging_kover4),("merging_kover8", merging_kover8)])

num_trials = 5
sigma = 1.0
k=16  # true number of pieces/rectangles of function
n_vals=[k*10,k*100,k*500]
d = 5         # dimension of samples
z = 2         # the number of dimensions the piecewise functions are defined in
mses = Dict{String, Array{Float64, 2}}()
times = Dict{String, Array{Float64, 2}}()
num_pieces = Dict{String, Array{Int, 2}}()

for algo_name in keys(algos)
    mses[algo_name] = Array{Float64,2}(undef, length(n_vals), num_trials)
    times[algo_name] = Array{Float64,2}(undef, length(n_vals), num_trials)
    num_pieces[algo_name] =  Array{Int64,2}(undef, length(n_vals), num_trials)
end

for i=1:length(n_vals)
    n_val = n_vals[i]
    for ii = 1:num_trials
        y, ystar, X = generate_random_regression_data(k, n_val, d, z, sigma)
        for (algo_name, algo_fun) in algos
            result = @timed algo_fun(X, y, k, n_val)
            yhat_result = result[1][1]
            pieces_output = result[1][2]
            time_elapsed = result[2]
            mses[algo_name][i, ii] = mse(yhat_result, ystar)
            times[algo_name][i, ii] = time_elapsed
            num_pieces[algo_name][i, ii] = pieces_output
        end
    end
end

mses_mean = Dict{AbstractString, Array{Float64, 1}}()
mses_std = Dict{AbstractString, Array{Float64, 1}}()
times_mean = Dict{AbstractString, Array{Float64, 1}}()
times_std = Dict{AbstractString, Array{Float64, 1}}()
pieces_mean = Dict{AbstractString, Array{Float64, 1}}()
pieces_std = Dict{AbstractString, Array{Float64, 1}}()
for algo_name in keys(mses)
    mses_mean[algo_name] = vec(mean(mses[algo_name], dims=2))
    mses_std[algo_name] = vec(std(mses[algo_name], dims=2))
    times_mean[algo_name] = vec(mean(times[algo_name], dims=2))
    times_std[algo_name] = vec(std(times[algo_name], dims=2))
    pieces_mean[algo_name] = vec(mean(num_pieces[algo_name], dims=2))
    pieces_std[algo_name] = vec(std(num_pieces[algo_name], dims=2))
end
