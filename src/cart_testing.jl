include("linear_merging_algorithm.jl")
include("synthetic_data_generation.jl")
using PyCall
using Statistics

import Conda
#Conda.add("scikit-learn")

using ScikitLearn
@sk_import tree: DecisionTreeRegressor

X = [[52, 34, 1, 4, 305, 1, 253],
 [78, 39, 1, 1, 382, 4, 304],
 [241, 34, 1, 4, 1127, 4, 886]]
y = [[5152], [5635], [23940]]

mt = DecisionTreeRegressor(max_leaf_nodes = 2)
mt.fit(X, y)
print(mt)
#print("Number of nodes: {}".format(mt.tree_.node_count))
yhat = mt.predict(X)

function merging_k(X::Array{Array{Float64,1},1}, y::Array{Float64,1}, k::Int, n::Int)
    levels = ceil(Int,log(2,n))
    leaves = fit_linear_merging(X, y, sigma, z, levels, k, convert(Float64,k))
    yhat = leaves_to_yhat(X,leaves) #reconstruct the yhat from leaves
    return yhat, length(leaves)
end
