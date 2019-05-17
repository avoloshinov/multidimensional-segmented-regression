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
    #push!(X, [row.temp, row.hum, row.atemp,row.windspeed]) #row.mnth
    #push!(y, row.cnt)
    #println("a=$(row.hr),a=$(row.hr), b=$(row.holiday), c=$(row.temp)")
    push!(X, [row.lstat,row.rm, row.crim,row.dis]) #row.mnth
    push!(y, row.medv)
end

function merging(X::Array{Array{Float64,1},1}, y::Array{Float64,1}, z::Int, k::Int, n::Int)
    stop_merge_param = k/4
    levels = ceil(Int,log(2,n))
    leaves = fit_linear_merging(X, y, 1.0, z, levels, k, convert(Float64,stop_merge_param))
    yhat = leaves_to_yhat(X,leaves) #reconstruct the yhat from leaves
    return yhat, length(leaves)
end

z=2
k=8
n=length(y)
println("num samples ", n)
yhat_merging, num_leaves_merging = merging(X,y,z,k,n)
err_merging = mse(yhat_merging,y)

println("num leaves merging ", num_leaves_merging)
println("mse merging ", err_merging)


regressor = DecisionTreeRegressor(max_leaf_nodes = 20)
regressor.fit(X, y)
yhat_cart = regressor.predict(X)

err_cart =mse(yhat_cart,y)
println("mse cart ", err_cart)
