include("linear_merging_algorithm.jl")
include("synthetic_data_generation_constant.jl")
using PyCall
using Statistics

import Conda
#Conda.add("scikit-learn")

using ScikitLearn
@sk_import tree: DecisionTreeRegressor

function merging_k(X::Array{Array{Float64,1},1}, y::Array{Float64,1}, z::Int, k::Int, n::Int)
    levels = ceil(Int,log(2,n))
    leaves = fit_linear_merging(X, y, sigma, z, levels, k, convert(Float64,k))
    yhat = leaves_to_yhat(X,leaves) #reconstruct the yhat from leaves
    return yhat, length(leaves)
end

function merging_kover2(X::Array{Array{Float64,1},1}, y::Array{Float64,1}, z::Int, k::Int, n::Int)
    levels = ceil(Int,log(2,n))
    leaves = fit_linear_merging(X, y, sigma, z, levels, k, convert(Float64,k/2))
    yhat = leaves_to_yhat(X,leaves) #reconstruct the yhat from leaves
    return yhat, length(leaves)
end

function merging_kover4(X::Array{Array{Float64,1},1}, y::Array{Float64,1}, z::Int, k::Int, n::Int)
    levels = ceil(Int,log(2,n))
    leaves = fit_linear_merging(X, y, sigma, z, levels, k, convert(Float64,k/4))
    yhat = leaves_to_yhat(X,leaves) #reconstruct the yhat from leaves
    return yhat, length(leaves)
end

function merging_kover8(X::Array{Array{Float64,1},1}, y::Array{Float64,1}, z::Int, k::Int, n::Int)
    levels = ceil(Int,log(2,n))
    leaves = fit_linear_merging(X, y, sigma, z, levels, k, convert(Float64,k/8))
    yhat = leaves_to_yhat(X,leaves) #reconstruct the yhat from leaves
    return yhat, length(leaves)
end

# function merging_kover16(X::Array{Array{Float64,1},1}, y::Array{Float64,1}, z::Int, k::Int, n::Int)
#     levels = ceil(Int,log(2,n))
#     leaves = fit_linear_merging(X, y, sigma, z, levels, k, convert(Float64,k/16))
#     yhat = leaves_to_yhat(X,leaves) #reconstruct the yhat from leaves
#     return yhat, length(leaves)
# end

function cart_trial(X::Array{Array{Float64,1},1}, y::Array{Float64,1},k::Int)
    regressor = DecisionTreeRegressor(max_leaf_nodes=k) #can do max_depth too
    regressor.fit(X, y)
    y_cart = regressor.predict(X)
    return [y_cart, regressor]
end

function run_experiments(num_trials::Int, sigma::Float64, k::Int, n_vals::Array{Int,1}, d::Int, z::Int)
    algos = Dict([("merging_k", merging_k), ("merging_kover2", merging_kover2), ("merging_kover4", merging_kover4),("merging_kover8", merging_kover8)])

    mses = Dict{String, Array{Float64, 2}}()
    times = Dict{String, Array{Float64, 2}}()
    num_pieces = Dict{String, Array{Int, 2}}()

    for algo_name in keys(algos)
        mses[algo_name] = Array{Float64,2}(undef, length(n_vals), num_trials)
        times[algo_name] = Array{Float64,2}(undef, length(n_vals), num_trials)
        num_pieces[algo_name] =  Array{Int64,2}(undef, length(n_vals), num_trials)
    end

    true_mses = Array{Float64,2}(undef, length(n_vals), num_trials)
    true_pieces = Array{Float64,2}(undef, length(n_vals), num_trials)

    cart1_mses = Array{Float64,2}(undef, length(n_vals), num_trials)
    cart1_pieces = Array{Float64,2}(undef, length(n_vals), num_trials)
    cart1_times = Array{Float64,2}(undef, length(n_vals), num_trials)
    cart2_mses = Array{Float64,2}(undef, length(n_vals), num_trials)
    cart2_pieces = Array{Float64,2}(undef, length(n_vals), num_trials)
    cart2_times = Array{Float64,2}(undef, length(n_vals), num_trials)


    @timed 30^30

    for i=1:length(n_vals)
        n_val = n_vals[i]
        for ii = 1:num_trials
            y, ystar, y_opt, X = generate_random_regression_data(k, n_val, d, z, sigma)

            true_mses[i, ii] = mse(y_opt,ystar)
            true_pieces[i,ii] = k

            #cart_result = @timed cart_trial(X,y,k)
            cart_result = @timed cart_trial(X,y,k)
            cart_result = @timed cart_trial(X,y,k)
            y_cart = cart_result[1][1]
            cart_time = cart_result[2]
            regressor = cart_result[1][2]

            cart1_times[i,ii] = cart_time
            cart1_mses[i, ii] = mse(y_cart,ystar)
            cart1_pieces[i,ii] = length(unique(regressor.apply(X)))

            cart_result = @timed cart_trial(X,y,Int(1.5*k))
            cart_result = @timed cart_trial(X,y,Int(1.5*k))
            y_cart = cart_result[1][1]
            cart_time = cart_result[2]
            regressor = cart_result[1][2]

            cart2_times[i,ii] = cart_time
            cart2_mses[i, ii] = mse(y_cart,ystar)
            cart2_pieces[i,ii] = length(unique(regressor.apply(X)))

            for (algo_name, algo_fun) in algos
                #result = @timed algo_fun(X, y, z, k, n_val)
                result = @timed algo_fun(X, y, z, k, n_val)
                result = @timed algo_fun(X, y, z, k, n_val)
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

    mses_mean["true fit"] = vec(mean(true_mses, dims=2))
    mses_std["true fit"] = vec(std(true_mses, dims=2))
    pieces_mean["true fit"] = vec(mean(true_pieces, dims=2))
    pieces_std["true fit"] = vec(std(true_pieces, dims=2))

    mses_mean["CART fit 16"] = vec(mean(cart1_mses, dims=2))
    mses_std["CART fit 16"] = vec(std(cart1_mses, dims=2))
    pieces_mean["CART fit 16"] = vec(mean(cart1_pieces, dims=2))
    pieces_std["CART fit 16"] = vec(std(cart1_pieces, dims=2))
    times_mean["CART fit 16"] = vec(mean(cart1_times, dims=2))
    times_std["CART fit 16"] = vec(std(cart1_times, dims=2))

    mses_mean["CART fit 24"] = vec(mean(cart2_mses, dims=2))
    mses_std["CART fit 24"] = vec(std(cart2_mses, dims=2))
    pieces_mean["CART fit 24"] = vec(mean(cart2_pieces, dims=2))
    pieces_std["CART fit 24"] = vec(std(cart2_pieces, dims=2))
    times_mean["CART fit 24"] = vec(mean(cart2_times, dims=2))
    times_std["CART fit 24"] = vec(std(cart2_times, dims=2))

    return mses_mean, mses_std, times_mean, times_std, pieces_mean, pieces_std
end
