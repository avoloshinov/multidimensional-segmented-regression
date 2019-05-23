# dataset info http://lib.stat.cmu.edu/datasets/boston
# dataset https://github.com/selva86/datasets/blob/master/BostonHousing.csv

include("linear_merging_algorithm.jl")
using ScikitLearn
@sk_import tree: DecisionTreeRegressor

using CSV
f = CSV.File("boston.csv")

X = Array{Array{Float64,1}}(undef,0)
y = Array{Float64,1}(undef,0)
for row in f
    push!(X, [row.lstat, row.rm,row.dis,row.crim, row.age, row.b,row.ptratio,row.nox,row.tax, row.zn,row.indus,row.chas,row.rad])
    push!(y, row.medv)
end

function merging(X::Array{Array{Float64,1},1}, y::Array{Float64,1}, z::Int, k::Int, n::Int, sigma::Float64)
    stop_merge_param = k
    levels = ceil(Int,log(2,n))
    leaves, root = fit_linear_merging(X, y, sigma, z, levels, k, convert(Float64,stop_merge_param))
    yhat = leaves_to_yhat(X,leaves) #reconstruct the yhat from leaves
    return yhat, length(leaves), root
end

function cart_trial(X::Array{Array{Float64,1},1}, y::Array{Float64,1},num_leaves_merging::Int)
    regressor = DecisionTreeRegressor(max_leaf_nodes = num_leaves_merging)
    regressor.fit(X, y)
    yhat_cart = regressor.predict(X)
    param = regressor.feature_importances_
    return regressor, yhat_cart, param
end

function find_best(sigma_vals::Array{Float64}, k_vals::Array{Int},z::Int)

    mses = Dict{String, Array{Float64, 2}}()
    mses["cart"] = Array{Float64,2}(undef, length(sigma_vals), length(k_vals))
    mses["merging"] = Array{Float64,2}(undef, length(sigma_vals), length(k_vals))

    best_sigma = sigma_vals[1]
    best_k = k_vals[1]
    cur_best_dif = 100000000
    for s=1:length(sigma_vals)
        for kk=1:length(k_vals)
            n=length(y)
            result = @timed merging(X,y,z,k_vals[kk],n,sigma_vals[s])
            yhat_merging = result[1][1]
            num_leaves_merging = result[1][2]
            time_merging = result[2]
            err_merging = mse(yhat_merging,y)
            cart_result = @timed cart_trial(X,y,num_leaves_merging)
            yhat_cart = cart_result[1][2]
            cart_time = cart_result[2]
            err_cart =mse(yhat_cart,y)
            mses["cart"][s, kk] = err_cart
            mses["merging"][s, kk] = err_merging
            dif = err_merging
            if dif<cur_best_dif
                best_k = k_vals[kk]
                best_sigma = sigma_vals[s]
                cur_best_dif = dif
            end
        end
    end
    return best_sigma, best_k, mses
end

z=3
sigma_vals = [1.0,2.0,3.0,4.0,5.0,10.0]
k_vals = [1,2,3,4,5,6]
sigma,k, mses = find_best(sigma_vals, k_vals,z)

sigma = 2.0
k=3
println("sigma is ", sigma)
println("k is ", k)

n=length(y)
println("num samples ", n)
result = @timed merging(X,y,z,k,n,sigma)
yhat_merging = result[1][1]
num_leaves_merging = result[1][2]
time_merging = result[2]
err_merging = mse(yhat_merging,y)

cart_result = @timed cart_trial(X,y,num_leaves_merging)
yhat_cart = cart_result[1][2]
cart_time = cart_result[2]
param = cart_result[1][3]
println(param)

err_cart =mse(yhat_cart,y)

println("num leaves merging ", num_leaves_merging)
println("mse merging ", err_merging)
println("mse cart ", err_cart)
println("time merging ", time_merging)
println("time cart ", cart_time)
