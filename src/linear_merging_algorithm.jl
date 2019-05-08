include("building_tree_structure.jl")

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

function fit_linear_merging(X::Array{Array{Float64,1},1}, y::Array{Float64,1}, sigma::Float64, z::Int, levels::Int, k::Int, upper_bound::Float64)
    root, leaves = create_tree(X, y, z,levels)
    #print_tree(root,levels,true)
    n = length(y)
    kp = k*(log(2,n)^2)

    # identify which leaves have all siblings as leaves as well
    # if a leaf has a parent that has all children has leaves, then this is a candidate merge
    candidate_pieces = find_candidate_set(leaves)


    #make sure the candidate set is large enough
    while length(candidate_pieces) > upper_bound#2*kp
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
        selected_threshold = max(ceil(Int, length(candidate_pieces) - upper_bound + 1))
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

function leaves_to_yhat(X::Array{Array{Float64,1},1}, leaves::Array{NodeRect,1})
    n = size(X)[1]
    yhat = Array{Float64,1}(undef,n)

    for leaf in leaves
        for index in leaf.data
            yhat[index] = sum(X[index] .* leaf.theta)
        end
    end

    return yhat
end
