include("matrix_manipulations_helper.jl")

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

# given a parent, splits the data into 2^z children/nodeRects
function add_children_to_parent(Y::Array{Array{Float64,1},1}, y::Array{Float64,1}, parent::NodeRect, z::Int, data_ordering:: Array{Array{Int,1},1})

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
                output = add_children_to_parent(X, y, cur_children[i],z,data_ordering)
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
