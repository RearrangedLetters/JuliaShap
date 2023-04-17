abstract type ShapleyValues end

struct MonteCarloShapley <: ShapleyValues
    values
    samples::Int
end