using LinearAlgebra
using SparseArrays
using JuMP
using MosekTools
using Plots

n = 101
w = 4 * pi

⊗ = kron

Δ = spdiagm(-1 => ones(n-1), 0 => -2*ones(n), 1 => ones(n-1))
A = (n*n) / (w*w) * (Δ ⊗ sparse(I, n, n) + sparse(I, n, n) ⊗ Δ)

left_idx = div(n, 4)
right_idx = n - left_idx + 1


z_2d = zeros(n, n)
z_2d[1:left_idx, left_idx:right_idx] .= 0
b_2d = zeros(n, n)
b_2d[1:left_idx, left_idx:right_idx] .= n^2

l = @layout [a b c]
p1 = heatmap(b_2d, title=raw"Excitation (S)", colorbar=:none)

z_hat = z_2d[:]
b = b_2d[:]

max_iter = 400

θ_max = 2*ones(n*n)
θ_min = 1*ones(n*n)

mask_2d = zeros(n, n)
mask_2d[end-div(n,4):end, left_idx:right_idx] .= 1
mask = mask_2d[:]

p2 = heatmap(mask_2d, title=raw"Mask (B)", colorbar=:none)

θ_bar = (θ_max + θ_min) / 2
ρ = (θ_max - θ_min) / 2

@time begin

# initial solve
sol = [
    Diagonal(mask) (A + Diagonal(θ_bar))';
    (A + Diagonal(θ_bar)) spzeros(n^2, n^2)
] \ [mask .* z_hat; b]

signs = 2 * (sol[1:n^2] .≥ 0) .- 1

ϵ = 1e-4
ϵ_tol = 1e-5

final_design = zeros(n, n)
final_field = nothing

old_val = Inf

# Field-based optimization
for curr_iter ∈ 1:max_iter
    global old_val, final_design

    @info "Currently on iteration $(curr_iter)"
    m = Model(with_optimizer(Mosek.Optimizer))
    @variable(m, z[1:n^2])
    @variable(m, y[1:n^2])
    @variable(m, t)

    @objective(m, Min, t)

    @constraint(m,[t; mask .* z] ∈ SecondOrderCone())
    @constraint(m, A * z + θ_bar .* z + ρ .* y .== b)
    @constraint(m, y .<= signs .* z)
    @constraint(m, -y .<= signs .* z)
    optimize!(m)

    @info "Terminated with objective value $(objective_value(m))"

    near_zero = abs.(value.(z)) .≤ ϵ
    if !any(near_zero) || old_val - objective_value(m) < ϵ_tol
        @info "No more directions to switch, breaking."
        final_design .= clamp.(reshape(value.(y) ./ value.(z), n, n), -1, 1)
        final_field = value.(z)
        break
    end

    old_val = objective_value(m)

    signs[near_zero] .= -signs[near_zero]
end

end

p3 = heatmap(final_design, title="Final design", colorbar=:none)
plot(p1, p2, p3, layout=l, size=(1200, 400))
savefig("paper_figures/complete_figure.pdf")

heatmap(reshape(final_field, n, n), colorbar=:none, size=(400, 400))
savefig("paper_figures/output_field.pdf")