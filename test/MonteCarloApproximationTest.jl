using Test
using Flux
using BSON: @load
using MLDatasets
using ImageIO
using ImageShow
using JLD: save, load
using Revise

using JuliaShap: shapley_values_monte


begin
    train_x_raw, train_y_raw = MNIST(split=:train)[:]
    save("train_x_raw.jld", "data", train_x_raw[:, :, 1:25])
    save("train_y_raw.jld", "data", train_y_raw[1:25])
end

@testset "MNIST Shapley values (1)" begin
    @load "test/models/model_MNIST.bson" model
    f(x) = model(x[:])
    train_x_raw = load("train_x_raw.jld")["data"]
    five = train_x_raw[:, :, 1]
    background = train_x_raw[:, :, 2:5]

    shapley_values_five = shapley_values_monte(f, five, 6, background, 2)
    @info shapley_values_five, size(first(shapley_values_five))
end