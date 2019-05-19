# dataset info http://lib.stat.cmu.edu/datasets/boston
# dataset https://github.com/selva86/datasets/blob/master/BostonHousing.csv

include("linear_merging_algorithm.jl")
using ScikitLearn
@sk_import tree: DecisionTreeRegressor

using CSV
f = CSV.File("day.csv")

X = Array{Array{Float64,1}}(undef,0)
y = Array{Float64,1}(undef,0)

for row in f
    push!(X, [row.mnth,row.temp,row.windspeed,row.hum])
    push!(y, row.cnt)
end

#X = X[1:50]
#y = y[1:50]

function merging(X::Array{Array{Float64,1},1}, y::Array{Float64,1}, z::Int, k::Int, n::Int)
    stop_merge_param = k
    levels = ceil(Int,log(2,n)+1)
    root, stuff = create_tree(X, y, z,levels)
    println(length(stuff))
    #print_leaves(stuff)
    leaves = fit_linear_merging(X, y, 0.0, z, levels, k, convert(Float64,stop_merge_param))
    yhat = leaves_to_yhat(X,leaves) #reconstruct the yhat from leaves
    return yhat, length(leaves)
end

z=2
k=6
n=length(y)
println("num samples ", n)
result = @timed merging(X,y,z,k,n)
yhat_merging = result[1][1]
num_leaves_merging = result[1][2]
time_merging = result[2]
err_merging = mse(yhat_merging,y)


function cart_trial(X::Array{Array{Float64,1},1}, y::Array{Float64,1})
    regressor = DecisionTreeRegressor(max_leaf_nodes = num_leaves_merging)
    regressor.fit(X, y)
    yhat_cart = regressor.predict(X)
    return regressor, yhat_cart
end

cart_result = @timed cart_trial(X,y)
yhat_cart = cart_result[1][2]
cart_time = cart_result[2]

err_cart =mse(yhat_cart,y)

println("num leaves merging ", num_leaves_merging)
println("mse merging ", err_merging)
println("mse cart ", err_cart)
println("time merging ", time_merging)
println("time cart ", cart_time)
