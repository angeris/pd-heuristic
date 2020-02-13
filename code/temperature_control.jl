module Heuristic

using LinearAlgebra
using SparseArrays
using JuMP
using MosekTools
using PyPlot

import Base.abs

# rc("text", usetex=true)

# --- Helper functions
function sum_squares(m, x)
    t = @variable(m)
    @constraint(m, [t+1; t-1; 2*x] ∈ SecondOrderCone())

    return t
end

function norm(m, x)
    t = @variable(m)
    @constraint(m, [t; x] ∈ SecondOrderCone())

    return t
end

function abs(v::VariableRef)
    model = v.model

    t = @variable(model)
    @constraint(model, t ≥ v)
    @constraint(model, t ≥ -v)

    return t
end
# --- End helpers

@time begin

n_edges = 3
n_vertices = 3

C = Diagonal([.5, .05])
B = .01 * Diagonal(ones(2))
A = [
    -1 0 1
    1 -1 0
    0 1 -1
]

g_min = ones(3)
g_max = 10 * g_min

g_bar = Diagonal((g_max + g_min) / 2)
ρ = Diagonal((g_max - g_min) / 2)

m = 200

t = range(0, 1, length=m)
ω = 4*π
T_min = 50
K = 20

e_ambient = K * (1 .+ sin.(ω * t)) .+ T_min

h = 1/m
ϵ = 1e-5
ϵ_tol = 1e-7

old_val = Inf

max_iter = 1000

e_min, e_max = 65, 75

# --- Initial solve
g_init = copy(g_bar)

model = Model(with_optimizer(Mosek.Optimizer))
@variable(model, e[1:n_vertices, 1:m])
@variable(model, u[1:2, 1:m-1])

# Objective and control constraints
@objective(model, Min, norm(model, u[:]) + 1e-4 * h^2 * sum_squares(model, e[1,1:m-1] .- e[1,2:m]))
@constraint(model, e_min .≤ e[1, :] .≤ e_max)
@constraint(model, e_min .≤ e[2, :] .≤ e_max)

# Model constraints
@constraint(model, e[1:2, 1] .== e[1:2, end])
@constraint(model, e[3, :] .== e_ambient)
@constraint(model, C * e[1:2, 2:m] .== C * e[1:2, 1:m-1] - h * A[1:2, :] * (g_init * A' * e[:, 1:m-1]) + h * B * u)

optimize!(model)

@show termination_status(model)

signs = sign.(A' * value.(e)[:, 1:m-1])

signs[signs .== 0] .= 1
# --- End initial solve



# Repeated optimization
for curr_iter ∈ 1:max_iter
    global old_val, final_design

    @info "Currently on iteration $(curr_iter)"
    model = Model(with_optimizer(Mosek.Optimizer))
    @variable(model, e[1:n_vertices, 1:m])
    @variable(model, u[1:2, 1:m-1])

    @variable(model, v[1:n_edges, 1:m-1])
    @variable(model, w[1:n_edges, 1:m-1])
    @variable(model, x[1:n_edges, 1:m-1])

    # Objective and control constraints
    @objective(model, Min, norm(model, u[:]) + 1e-4 * h^2 * sum_squares(model, e[1,1:m-1] .- e[1,2:m]))
    @constraint(model, e_min .≤ e[1, :] .≤ e_max)
    @constraint(model, e_min .≤ e[2, :] .≤ e_max)

    # Model constraints
    @constraint(model, e[1:2, 1] .== e[1:2, end])
    @constraint(model, e[3, :] .== e_ambient)
    @constraint(model, C * e[1:2, 2:m] .== C * e[1:2, 1:m-1] - h * A[1:2, :] * w + h * B * u)
    @constraint(model, v .== A' * e[:,1:m-1])
    @constraint(model, w .== g_bar * v + ρ * x)
    @constraint(model, x .<= signs .* v)
    @constraint(model, -x .<= signs .* v)

    optimize!(model)

    @info "Terminated with objective value $(objective_value(model))"
    @info primal_status(model)

    # @show value.(z)

    near_zero = abs.(value.(v)) .≤ ϵ

    if !any(near_zero) || old_val - objective_value(model) < ϵ_tol
        if !any(near_zero)
            @info "No more directions to switch, breaking."
        else
            @info "Tolerance reached, breaking."
        end

	# Plot figures
        figure(figsize=(5, 15))

        subplot(311)
        title("Temperatures (e)")
        plot(1:m, value.(e[1,:]), "--", label="Room 1")
        plot(1:m, value.(e[2,:]), label="Room 2")
        plot(1:m, value.(e[3,:]), "-.", label="Outside")
        axhline(65, ls=":", color="k")
        axhline(75, ls=":", color="k")
        legend()

        subplot(312)
        title("Conductances (g)")
        g_fix = value.(x) ./ value.(v)
        g_fix[abs.(value.(v)) .≤ ϵ] .= -1
        g_fix .*= ρ[1]
        g_fix .+= g_bar[1]

        g_1 = g_fix[1,:]
        g_2 = g_fix[2,:]
        g_3 = g_fix[3,:]
        step(1:m-1, g_1, label="Room 1 - Room 2", ls="--")
        step(1:m-1, g_2, label="Room 2 - Outside", ls=":")
        step(1:m-1, g_3, label="Room 1 - Outside")
        legend()

        subplot(313)
        title("Input control (u)")
        u_fix = value.(u)
        u_fix[abs.(u_fix) .≤ ϵ] .= 0
        plot(1:m-1, u_fix[1,:], "--", label="Room 1 input")
        plot(1:m-1, u_fix[2,:], ":", label="Room 2 input")
        legend()

        savefig("new_difference_control.pdf", bbox_inches="tight")

        close("all")
        break
    end

    old_val = objective_value(model)

    signs[near_zero] .= -signs[near_zero]
end

end

end
