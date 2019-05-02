using LinearAlgebra

mutable struct NodeRect
    data::Array{Int, 1}                 # the indices of X that belong in this rectangle
    indices::Array{Array{Float64,1},1}  # indices of the first z dimensions of the rectangle
    theta::Array{Float64,1}             # computed theta value for these points
    children::Array{Any,1}              # either an array of NodeRects, or [] if leaf
    parent::Any                         # 0 is root, a NodeRect otherwise
end

function is_leaf(node::NodeRect)
    return isempty(node.children)
end

function print_leaves(leaves::Array{NodeRect,1})
    for leaf in leaves
        println("leaf data is ", leaf.data)
        println("indices are", leaf.indices)
        println()
    end
end

function print_tree(root::NodeRect, levels::Int, print_empty_nodes::Bool)
    cur_children = [root]
    cur_children_temp = Array{NodeRect,1}(undef,0)

    for level=0:levels
            println("Level ", level)
            println()
        for i=1:size(cur_children)[1]
            cur = cur_children[i]
            if print_empty_nodes || cur.data != []
                println("data is ", cur.data)
                println("indices are ", cur.indices)
                println("theta is ", cur.theta)
                println("is leaf ", is_leaf(cur))
                for child in cur_children[i].children
                    push!(cur_children_temp, child)
                end
                println()
            end
        end
        cur_children = cur_children_temp
        cur_children_temp = Array{NodeRect,1}(undef,0)
    end
end

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

# fits a rectangular piece to the data_indices selected from X.
function new_node(data_indices::Array{Int,1}, X::Array{Array{Float64,1},1}, y::Any, indices::Array{Array{Float64,1},1}, parent::Any)
    Y = indices_to_sub_matrix(X,data_indices)
    Z = change_to_matrix_format(Y)

    y_new = indices_to_sub_labels(y,data_indices)

    if y_new != []
        theta = Z \ y_new
    else
        theta = []
    end
    return NodeRect(data_indices,indices,theta,[],parent)
end

# outputs a dictionary where the keys are the possible indices
# and the values are empty (to be filled with indices of corresponding data points)
function construct_current_rectangles(z::Int, new_index_bounds::Array{Array{Array{Float64,1},1},1})
    flag=0
    count = 2^z
    the_bit_list = Array{Int}(undef,count, z)
    for i=1 : z
        index=1
        count = count/2
        while index < 2^z + 1
            for j=1 : count
                the_bit_list[index,i]=flag
                index=index+1
            end
            if flag==0
                flag=1
            else
                flag=0
            end
        end
    end

    all_indices = Dict{Array{Array{Float64,1},1},Array{Int64,1}}()

    for i=1 : 2^z
        #define the indices of one rectangle [[dim1-left,dim1-right],[dim2-left,dim2-right],etc]
        indices = Array{Array{Float64,1},1}(undef,0)
        for j=1 : z
            if the_bit_list[i,j]==0
                push!(indices,new_index_bounds[j][1])
            else
                push!(indices, new_index_bounds[j][2])
            end
        end
        all_indices[indices]=[]
    end

    return all_indices
end

# ranks the first z coordinates of X
function coordinate_ranking(X::Array{Array{Float64,1},1},z::Int)
    cols = transform_matrix(X)
    rankings = Array{Array{Int,1},1}(undef, 0)
    n=size(X)[1]
    for row=1:n
        rank = Array{Int,1}(undef,0)
        for i=1:z
            index = findfirst(x->x==X[row][i],sort(cols[i]))
            push!(rank, index)
        end
        push!(rankings,rank)
    end
    return rankings
end

# given a parent, splits the data into 2^z children/nodeRects
function add_children_to_parent(Y::Array{Array{Float64,1},1}, parent::NodeRect, z::Int, data_ordering:: Array{Array{Int,1},1})

    X = indices_to_sub_matrix(Y,parent.data)
    n = length(parent.data)

    #divide each dimension in half to prepare new dimensions for children
    new_index_bounds = Array{Array{Array{Float64,1},1},1}(undef,0)
    for i=1 : z
        left = parent.indices[i][1]
        right = parent.indices[i][2]
        middle = (left+right)/2
        push!(new_index_bounds,[[left,middle],[middle,right]])
    end

    # maps indicies/bounds of rectangle to the indices of points in that rectangle
    all_indices = construct_current_rectangles(z,new_index_bounds)

    # for each data point, decide which child it belongs in
    for i in parent.data
        data_ind = data_ordering[i]
        indices = Array{Array{Float64,1},1}(undef,0)
        for j=1:z
            cur = data_ind[j]
            low = new_index_bounds[j][1][1]
            mid = new_index_bounds[j][1][2]
            upper = new_index_bounds[j][2][2]

            if low <= cur && cur < mid
                push!(indices, [low,mid])
            else
                push!(indices, [mid,upper])
            end
        end
        push!(all_indices[indices], i)
    end

    for (indices, x_indices) in all_indices
        if x_indices != []
            push!(parent.children, new_node(x_indices, Y, y, indices, parent))
        end
    end

    return [parent,parent.children]
end


function create_tree(X::Array{Array{Float64,1},1}, y::Array{Float64,1}, z::Int, levels::Int)
    n = length(y)
    indices = fill(float([0,n+1]),z)
    root = new_node(collect(1:n), X, y, indices,0)
    cur_children = Array{NodeRect,1}(undef,0)
    cur_children = [root]
    cur_children_temp = Array{NodeRect,1}(undef,0)

    #the ordering of the first z dimensions of the data points
    data_ordering = coordinate_ranking(X,z)

    all_leaves = Array{NodeRect,1}(undef,0)

    for level=1:levels
        for i=1:size(cur_children)[1]
            if length(cur_children[i].data) > 1
                output = add_children_to_parent(X, cur_children[i],z,data_ordering)
                cur_children[i] = output[1]
                for child in output[2]
                    push!(cur_children_temp, child)
                end
            else
                push!(all_leaves, cur_children[i])
            end
        end
        cur_children = cur_children_temp
        cur_children_temp = Array{NodeRect,1}(undef,0)
    end

    return root, all_leaves
end

#even number of points in each true rectange?
function generate_random_regression_data(num_rectangles::Int, n::Int, d::Int, z::Int, sigma::Float64)
    X = randn(n, d)
    Y = change_to_array_format(X)
    data_ordering = coordinate_ranking(Y,z)
    # todo matrix to array of arrays for input

    num_intervals_per_dim = floor(Int,num_rectangles^(1/z))
    num_per_rect = floor(n/num_rectangles)

    ystar = Array{Float64,1}(undef,n)

    #all_intervals = Array{Array{Array{Float64, 1},1},1}(undef,0)

    intervals_1dim = Array{Array{Int64,1},1}(undef,0)
    lower = 0
    num_points_per_slice = floor(n/num_intervals_per_dim)
    upper = num_points_per_slice
    for i=1:num_intervals_per_dim
        push!(intervals_1dim,[lower,upper])
        lower=upper
        upper=upper+num_points_per_slice
    end

    N = floor(Int,num_intervals_per_dim)
    all_combos_digits = reverse.(digits.(0:N^N-1,base=N,pad=N)) #Array{Array{Int64,1},1}_

    # construct dictionaries where the indicies of intervals in each dimensions are the keys
    # values will be the indices of the X values that belong there
    all_rects = Dict{Array{Array{Int,1},1},Array{Int,1}}()
    for i=1:size(all_combos_digits)[1]
        one_rect = Array{Array{Int,1},1}(undef,0)
        for j=1:num_intervals_per_dim
            ind = all_combos_digits[i][j]+1
            push!(one_rect,intervals_1dim[ind])
        end
            all_rects[one_rect]=[]
    end

    for i=1:n
        data_ind = data_ordering[i]
        indices = Array{Array{Float64,1},1}(undef,0)
        for j=1:z
            cur = data_ind[j]
            h=1
            while h<= num_intervals_per_dim
                low = intervals_1dim[h][1]
                high = intervals_1dim[h][2]
                if low <= cur && cur < high
                    push!(indices, [low,high])
                    h=num_intervals_per_dim+1
                elseif (h==num_intervals_per_dim)
                    push!(indices, [low,high])
                end
                h=h+1
            end
        end
        push!(all_rects[indices], i)
    end


    for (indices, x_ind_list) in all_rects
        beta = 2 * rand(Float64, d) + ones(d)

        Y = Array{Array{Float64,1},1}(undef,0)
        for q in x_ind_list
            push!(Y, X[q,:])
        end

        if Y != []
            Z = change_to_matrix_format(Y)
            labels = vec(Z * beta)
            for q=1:size(x_ind_list)[1]
                ystar[x_ind_list[q]]=labels[q]
            end
        end
    end

    y = ystar + sigma * randn(n)
    return y, ystar, change_to_array_format(X)
end


########################################################

# is this correct
function rectangle_size(p::NodeRect)
    return length(p.data)
end

function rectangle_piece_merging_error(p::NodeRect, X::Array{Array{Float64,1},1}, y::Array{Float64,1}, sigma::Float64)
    new_y = indices_to_sub_labels(y, p.data)
    new_X =indices_to_sub_matrix(X, p.data)
    error = norm(new_y - change_to_matrix_format(new_X) * p.theta)^2
    return error - rectangle_size(p) * sigma^2
end

function mse(yhat, ystar)
   return (1.0 / length(yhat)) * sum((yhat - ystar).^2)
end

function find_candidate_set(leaves::Array{NodeRect,1})
    candidate_pieces = Array{NodeRect,1}(undef,0)
    for leaf in leaves
        parent = leaf.parent
        if !(parent in candidate_pieces)
            all_children_leaves = true
            for child in parent.children
                if !is_leaf(child)
                    all_children_leaves = false
                end
            end
            if all_children_leaves
                push!(candidate_pieces, parent)
            end
        end
    end
    return candidate_pieces
end

function fit_linear_merging(X::Array{Array{Float64,1},1}, y::Array{Float64,1}, sigma::Float64, z::Int, levels::Int, k::Int)
    root, leaves = create_tree(X, y, z,levels)
    #print_tree(root,levels,true)
    n = length(y)
    kp = k*(log(2,n)^2)

    # identify which leaves have all siblings as leaves as well
    # if a leaf has a parent that has all children has leaves, then this is a candidate merge
    candidate_pieces = find_candidate_set(leaves)


    #make sure the candidate set is large enough
    while length(candidate_pieces) > 2*kp
        #println("merging happening")
        #println("candidate length", length(candidate_pieces))
        #println("needed size", 2*kp)

        # for each candidate find the error
        candidate_errors = Array{Float64,1}(undef,0)
        for piece in candidate_pieces
            err = rectangle_piece_merging_error(piece, X, y, sigma)
            push!(candidate_errors,err)
        end

        # then find the set of parents with the largest errors
        # Select the num_holdout_pieces'th largest error (counting from the largest
        # error down) as threshold.
        selected_threshold = ceil(Int, length(candidate_pieces) - 2*kp + 1)
        error_threshold = (sort(candidate_errors))[selected_threshold]

        leaves_to_remove = Array{NodeRect,1}(undef,0)

        # and for each of those make their kids into [] and make them into leaf nodes
        for i=1:length(candidate_errors)
            if candidate_errors[i] <= error_threshold
                new_leaf_node = candidate_pieces[i]
                for child in new_leaf_node.children
                    push!(leaves_to_remove, child)
                end
                new_leaf_node.children = []
                push!(leaves,new_leaf_node)
            end
        end

        # remove the leaves that were merged from the leaves
        #println("size of removing", length(leaves_to_remove))
        #println("size of leaves currently", length(leaves))
        leaves = filter(x -> !(x in leaves_to_remove), leaves)
        candidate_pieces = find_candidate_set(leaves)
    end

    return leaves
end
