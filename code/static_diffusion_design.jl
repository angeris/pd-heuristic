module HeatDesignHeuristic

import Cairo, Fontconfig

using JuMP, MosekTools
using LightGraphs
using LinearAlgebra
using GraphPlot
using Compose

function initial_circuit(A, s, g_bar, ground_idx)
    L = g_bar * A * A'

    L[ground_idx, :] .= 0
    L[:, ground_idx] .= 0
    L[ground_idx, ground_idx] = 1

    s_copy = copy(s)
    s_copy[ground_idx] = 0

    return L \ s_copy
end

for m ∈ [11, 51]
    @time begin
    @info "Starting for grid graph with m=$m"
    graph = grid([m, m])

    A = incidence_matrix(graph, oriented=true)

    g_min, g_max = 1, 10
    g_bar = (g_max + g_min) / 2
    ρ = (g_max - g_min) / 2

    s = zeros(nv(graph))
    s[1] = -1
    s[end] = 1
    ground_idx = 1

    e_init = initial_circuit(A, s, g_bar, ground_idx)
    curr_signs = 2 * (A' * e_init .≥ 0) .- 1

    c_2d = zeros(m, m)
    side_length = div((m-1),4)
    c_2d[side_length:3*side_length, side_length:3*side_length] .= 1
    c = c_2d[:] / sum(c_2d)

    max_iter = 100
    ϵ_tol = 1e-6
    ϵ_terminate = 1e-5

    prev_iter_value = Inf

    for curr_iter ∈ 1:max_iter
        m = Model(with_optimizer(Mosek.Optimizer))

        @variable(m, e[1:nv(graph)])
        @variable(m, v[1:ne(graph)])
        @variable(m, w[1:ne(graph)])
        @variable(m, x[1:ne(graph)])

        @objective(m, Min, c' * e)

        @constraint(m, v .== A' * e)
        @constraint(m, A * w .== s)
        @constraint(m, w .== g_bar * v + ρ * x)
        @constraint(m, x .≤ curr_signs .* v)
        @constraint(m, -x .≤ curr_signs .* v)
        @constraint(m, e[ground_idx] .== 0)

        optimize!(m)

        @info "Finished optimization, current objective value $(objective_value(m)) at iteration $(curr_iter)"
        @info "Termination status: $(termination_status(m))."

        small_idx = abs.(value.(v)) .≤ ϵ_tol

        if !any(small_idx)
            @info "No signs left to change, breaking."
            break
        elseif prev_iter_value - objective_value(m) ≤ ϵ_terminate
            @info "Not decreasing fast enough, terminating."
            break
        end

        @info "Changing $(sum(small_idx)) signs."
        curr_signs[small_idx] .= -curr_signs[small_idx]
        prev_iter_value = objective_value(m)
    end
    end
end

end
