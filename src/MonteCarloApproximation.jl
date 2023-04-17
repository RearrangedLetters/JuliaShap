using Random

"""
An approximation of the shapley values using a Monte Carlo approach. This implementation is based on
the description here christophm.github.io/interpretable-ml-book/shapley.html.
"""

"""
f is the function used for prediction, e.g. the model
x is the instance to be explained
k is the class for which Shapley values are being calculated
M is the data from which samples are drawn (the background)
n is the number of samples

The result is an instance of MonteCarloShapley that stores the shapley values and remembers the
number of samples used. This information can be used to improve the approximation incrementally.
"""
function shapley_values_monte(f, x, M, n)
    @info "newnewnew"
    shapley_values = zeros((size(x)..., 10))
    number_of_entries = foldl(*, size(x))
    size_of_background = length(M[1, 1, :])
    n_rows = size(x)[1]
    n_cols = size(x)[2]
    for _ ∈ 1:n
        σ = reshape(randperm(number_of_entries), size(x))
        xₒ  = x[σ]
        zₒ  = M[:, :, rand(1:size_of_background)][σ]
        x₊ⱼ = copy(zₒ)
        x₋ⱼ = copy(zₒ)
        for j in 1:(number_of_entries - 1)
            x₊ⱼ[j] = xₒ[j]
            x₋ⱼ[j + 1] = xₒ[j + 1]
            row = div(j - 1, n_cols) + 1
            col = rem(j - 1, n_cols) + 1
            shapley_values[row, col, :] += f(x₊ⱼ) - f(x₋ⱼ)
        end
    end
    shapley_values ./= n
    return shapley_values, n
end

""" Notes
These StackedViews don't work, some stuff about indexing.
x₊ⱼ = reshape(StackedView(
                view(xₒ, 1:j),
                view(zₒ, (j + 1):number_of_entries)),
              size(x))
x₋ⱼ = reshape(StackedView(
                view(xₒ, 1:(j + 1)),
                view(zₒ, (j + 2):number_of_entries)),
              size(x))

In any case, it is not clear, whether views are preferable over copies.
If copies turn out to slow, replace them with a custom view struct.

"""