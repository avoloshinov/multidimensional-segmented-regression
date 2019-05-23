include("matrix_manipulations_helper.jl")

function combos(interval_options::Array{Array{Int,1},1}, z::Int)
    n = length(interval_options)
    all_rects = Dict{Array{Array{Int,1},1},Array{Int,1}}()
    cur = fill(1,z)
    final_cur = fill(n,z)

    notdone = true
    while notdone
        one_rect = Array{Array{Int,1},1}(undef,0)
        for a in cur
            push!(one_rect,interval_options[a])
        end
        all_rects[one_rect]=[]

        if cur == final_cur
            notdone = false
        end

        i=1
        while i<=z
            if cur[i]<n
                cur[i]=cur[i]+1
                i=z+1
            else #cur[i]==z
                cur[i]=1
            end
            i=i+1
        end
    end
    return all_rects
end

#even number of points in each true rectange?
function generate_random_regression_data(num_rectangles::Int, n::Int, d::Int, z::Int, sigma::Float64)
    X = randn(n, d)
    Y = change_to_array_format(X)
    data_ordering = coordinate_ranking(Y,z)

    num_intervals_per_dim = ceil(Int,num_rectangles^(1/z))
    # println(num_intervals_per_dim)
    #
    # if num_intervals_per_dim != num_rectangles^(1/z)
    #     error("can't partion into equal size rectangles with this parameter of num_rectangles")
    # end

    num_per_rect = floor(Int,n/num_rectangles)

    ystar = Array{Float64,1}(undef,n)

    #all_intervals = Array{Array{Array{Float64, 1},1},1}(undef,0)

    intervals_1dim = Array{Array{Int64,1},1}(undef,0)
    lower = 0
    num_points_per_slice = floor(Int,n/num_intervals_per_dim)
    upper = num_points_per_slice
    for i=1:num_intervals_per_dim
        push!(intervals_1dim,[lower,upper])
        lower=upper
        upper=upper+num_points_per_slice
    end

    all_rects = combos(intervals_1dim,z)

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

    y_opt = Array{Float64,1}(undef,n)

    for (indices, x_ind_list) in all_rects

        Y = Array{Array{Float64,1},1}(undef,0)
        y_cur = Array{Float64,1}(undef,0)
        for q in x_ind_list
            push!(Y, X[q,:])
            push!(y_cur,y[q])
        end

        if Y != []
            Z = change_to_matrix_format(Y)
            theta = pinv(Z)*y_cur
            labels = vec(Z * theta)

            for q=1:size(x_ind_list)[1]
                y_opt[x_ind_list[q]]=labels[q]
            end
        end
    end

    return y, ystar, y_opt, change_to_array_format(X)
end
