using LinearAlgebra

# from array of arrays to a 2d array
function change_to_matrix_format(X::Array{Array{Float64,1},1})
    return transpose(hcat(X...))
end

# from 2d array to array of arrays
function change_to_array_format(X::Array{Float64,2})
    Y = Array{Array{Float64,1},1}(undef,0)
    for row=1:size(X)[1]
        push!(Y,X[row,:])
    end
    return Y
end

# from array of arrays of rows to array of arrays of columns
function transform_matrix(X::Array{Array{Float64,1},1})
    return [hcat(X...)[i, :] for i in 1:size(hcat(X...), 1)]
end

# from full X matrix to the subset of only the target indices
function indices_to_sub_matrix(X::Array{Array{Float64,1},1}, target_indices::Array{Int,1})
    Y = Array{Array{Float64,1},1}(undef,0)
    for val in target_indices
        push!(Y,X[val])
    end
    return Y
end

# from full y labels to the subset of only the target labels
function indices_to_sub_labels(y::Array{Float64,1}, target_indices::Array{Int,1})
    y_new = Array{Float64,1}(undef,0)
    for val in target_indices
        push!(y_new,y[val])
    end
    return y_new
end

# ranks the first z coordinates of X
function coordinate_ranking(X::Array{Array{Float64,1},1},z::Int)
    cols = transform_matrix(X)
    rankings = Array{Array{Int,1},1}(undef, 0)
    n=size(X)[1]
    for row=1:n
        rank = Array{Int,1}(undef,0)
        for i=1:z
            index = findfirst(x->x==X[row][i],sort(unique(cols[i])))
            push!(rank, index)
        end
        push!(rankings,rank)
    end
    return rankings
end
