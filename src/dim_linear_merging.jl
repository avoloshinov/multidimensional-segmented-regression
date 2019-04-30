using LinearAlgebra

struct NodeRect
    data::Array{Array{Float64,1},1}     # all the points that belong in this node / in this rectangle
    indices::Array{Array{Float64,1},1}  # indices of the first z dimensions of the rectangle
    theta::Array{Float64,1}
    children::Array{Any,1}              # either an array of NodeRects, or [] if leaf
    parent::Any                         # 0 is root, a NodeRect otherwise
end

function is_leaf(node::NodeRect)
    return isempty(node.children)
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

function change_to_matrix_format(X::Array{Array{Float64,1},1})
    return transpose(hcat(X...))
end

function change_to_array_format(X::Array{Float64,2})
    Y = Array{Array{Float64,1},1}(undef,0)
    for row=1:size(X)[1]
        push!(Y,X[row,:])
    end
    return Y
end

# array of arrays of rows -> array of arrays of columns
function transform_matrix(X::Array{Array{Float64,1},1})
    return [hcat(X...)[i, :] for i in 1:size(hcat(X...), 1)]
end

# fit-rectangular piece?
function new_node(data::Array{Array{Float64,1},1}, y::Any, indices::Array{Array{Float64,1},1}, parent::Any)
    X = change_to_matrix_format(data)
    if y != []
        theta = X \ y
    else
        theta = []
    end
    return NodeRect(data,indices,theta,[],parent)
end

# outputs a dictionary where the keys are the possible indices
# and the values are empty (to be filled with data points)
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

    all_indices = Dict{Array{Array{Float64,1},1},Array{Array{Float64,1}}}()

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
    dict = Dict{Array{Float64,1},Array{Int,1}}()
    for row=1:n
        rank = Array{Int,1}(undef,0)
        for i=1:z
            index = findfirst(x->x==X[row][i],sort(cols[i]))
            push!(rank, index)
        end
        dict[X[row]] = rank
    end
    return dict
end

# given a parent, splits the data into 2^z children / nodeRects
function add_children_to_parent(parent::NodeRect, z::Int, data_ordering::Dict{Array{Float64,1},Array{Int64,1}})

    X = parent.data
    n = size(X)[1]

    #divide each dimension in half to prepare new dimensions for children
    new_index_bounds = Array{Array{Array{Float64,1},1},1}(undef,0)
    for i=1 : z
        left = parent.indices[i][1]
        right = parent.indices[i][2]
        middle = (left+right)/2
        push!(new_index_bounds,[[left,middle],[middle,right]])
    end

    # maps indicies of rectangle to points in that rectangle
    all_indices = construct_current_rectangles(z,new_index_bounds)
    all_indices_y_vals = construct_current_rectangles(z,new_index_bounds)

    # for each data point, decide which child it belongs in
    for i=1:n
        data_ind = data_ordering[X[i]]
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
        push!(all_indices[indices], X[i])
        push!(all_indices_y_vals[indices],[y[i]])
    end

    for (indices, x_vals) in all_indices
        y =vcat(all_indices_y_vals[indices]...)
        push!(parent.children, new_node(x_vals, y, indices, parent))
    end

    return [parent,parent.children]
end


function create_tree(X::Array{Array{Float64,1},1}, y::Array{Float64,1}, z::Int, levels::Int)
    n = float(size(X)[1])
    indices = fill([0,n+1],z)
    root = new_node(X, y, indices,0)
    cur_children = Array{NodeRect,1}(undef,0)
    cur_children = [root]
    cur_children_temp = Array{NodeRect,1}(undef,0)

    #the ordering of the first z dimensions of the data points
    data_ordering = coordinate_ranking(X,z)

    for level=1:levels
        for i=1:size(cur_children)[1]
            output = add_children_to_parent(cur_children[i],z,data_ordering)
            cur_children[i] = output[1]
            for child in output[2]
                push!(cur_children_temp, child)
            end
        end
        cur_children = cur_children_temp
        cur_children_temp = Array{NodeRect,1}(undef,0)
    end

    return root
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
        data_ind = data_ordering[X[i,:]]
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
  return size(X)[1]
end

function rectangle_piece_merging_error(p::NodeRect, X::Array{Float64,2}, y::Array{Float64,1}, sigma::Float64)
  return rectangle_piece_error(p, X, y) - piece_length(p) * sigma^2
end

# this needs to be the correct y tho... todo.....
function recangle_piece_error(p::NodeRect, y::Array{Float64,1})
  return norm(y - change_to_matrix_format(p.data) * p.theta)^2
end

#function linear_fit_error(X::Array{Float64,2}, y::Array{Float64,1}, left_index::Int, right_index::Int)
#  p = fit_linear_piece(X, y, left_index, right_index)
#  return linear_piece_error(p, X, y)
#end

function mse(yhat, ystar)
   return (1.0 / length(yhat)) * sum((yhat - ystar).^2)
end


##################
#
# function partition_to_vector(X::Array{Float64,2}, pieces::Array{LinearPiece,1})
#   n = pieces[end].right_index
#   (rows, cols) = size(X)
#   if n != rows
#     error("number of rows and rightmost index must match")
#   end
#   y = Array{Float64}(undef, n)
#   for ii = 1 : length(pieces)
#     p = pieces[ii]
#     y[p.left_index : p.right_index] = X[p.left_index : p.right_index, :] * p.theta
#   end
#   return y
# end
#
#
#
# function fit_linear_merging(X::Array{Float64,2}, y::Array{Float64,1}, sigma::Float64, num_target_pieces::Int, num_holdout_pieces::Int; initial_merging_size::Int=-1)
#
#   n = length(y)
#   if initial_merging_size <= 0
#     initial_merging_size = Int(sqrt(sqrt(n)))
#   end
#
#   # Initial partition
#   cur_pieces = Array{LinearPiece}(undef, 0)
#   num_remaining = n
#   cur_left = 1
#   while num_remaining > 0
#     cur_right = min(cur_left + initial_merging_size - 1, n)
#     num_remaining -= initial_merging_size
#     tmp_piece = fit_linear_piece(X, y, cur_left, cur_right)
#     push!(cur_pieces, tmp_piece)
#     cur_left = cur_right + 1
#   end
#   prev_pieces = Array{LinearPiece}(undef,0)
#
#   while length(cur_pieces) > num_target_pieces && length(cur_pieces) != length(prev_pieces)
#     prev_pieces = cur_pieces
#     cur_pieces = Array{LinearPiece}(undef, 0)
#
#     # Create a list of merging candidates and compute their errors
#     candidate_pieces = Array{LinearPiece}(undef,0)
#     candidate_errors = Array{Float64}(undef, 0)
#     for ii = 1:floor(Int, length(prev_pieces) / 2)
#       left_piece = prev_pieces[2 * ii - 1]
#       right_piece = prev_pieces[2 * ii]
#       new_piece = fit_linear_piece(X, y, left_piece.left_index, right_piece.right_index)
#       new_error = linear_piece_merging_error(new_piece, X, y, sigma)
#       push!(candidate_pieces, new_piece)
#       push!(candidate_errors, new_error)
#     end
#
#     # For an odd number of pieces, we directly include the last piece as a
#     # candidate.
#     if length(prev_pieces) % 2 == 1
#       last_piece = prev_pieces[end]
#       last_error = linear_piece_merging_error(last_piece, X, y, sigma)
#       push!(candidate_pieces, last_piece)
#       push!(candidate_errors, last_error)
#     end
#
#     # Select the num_holdout_pieces'th largest error (counting from the largest
#     # error down) as threshold.
#     selected_threshold = max(0, length(candidate_pieces) - num_holdout_pieces + 1)
#     error_threshold = (sort(candidate_errors))[selected_threshold]
#
#
#     # Count how many of the intervals are exactly at the threshold to avoid
#     # corner cases.
#     num_at_threshold = 0
#     num_above_threshold = 0
#     for ii = 1:length(candidate_pieces)
#       if candidate_errors[ii] == error_threshold
#         num_at_threshold += 1
#       elseif candidate_errors[ii] > error_threshold
#         num_above_threshold += 1
#       end
#     end
#     num_at_threshold_to_include = num_holdout_pieces - num_above_threshold
#
#     # Form the new partition
#     for ii = 1:length(candidate_pieces)
#       # Use the merge candidate if the error is small enough.
#       if candidate_errors[ii] < error_threshold
#         push!(cur_pieces, candidate_pieces[ii])
#       # If the error is exactly at the threshold, we make a special check
#       elseif candidate_errors[ii] == error_threshold && num_at_threshold_to_include > 0
#         num_at_threshold_to_include -= 1
#         push!(cur_pieces, candidate_pieces[ii])
#       else
#         # Otherwise, include the original pieces
#         push!(cur_pieces, prev_pieces[2 * ii - 1])
#         # Corner case for the last candidate piece, which might not be the
#         # result of a merge
#         if 2 * ii <= length(prev_pieces)
#           push!(cur_pieces, prev_pieces[2 * ii])
#         end
#       end
#     end
#   end
#
#   return cur_pieces
# end
