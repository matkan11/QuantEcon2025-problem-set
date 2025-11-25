#problem 3
# given

using Plots

#parameters of the CES utilities
sigma_low  = 0.2
sigma_high = 5.0

#endowments:, agent 1: (ω11, ω12) = (1, 1) and agent 2: (ω21, ω22) = (0.5, 1.5)
ω11, ω12 = 1.0, 1.0
ω21, ω22 = 0.5, 1.5

#total endowments
ω1_total = ω11 + ω21
ω2_total = ω12 + ω22

#demand for good 1 of a single agent
function demand_good1(p1, α, σ, ω1, ω2; p2 = 1.0)
    m = p1 * ω1 + p2 * ω2                       
    den = α^σ * p1^(1 - σ) + (1 - α)^σ * p2^(1 - σ)
    return α^σ * p1^(-σ) / den * m
end

#demand for good 2 of a single agent
function demand_good2(p1, α, σ, ω1, ω2; p2 = 1.0)
    m = p1 * ω1 + p2 * ω2
    den = α^σ * p1^(1 - σ) + (1 - α)^σ * p2^(1 - σ)
    return (1 - α)^σ * p2^(-σ) / den * m
end

#excess demand for good 1 for given (p1, x, σ)
#α1 = x, α2 = 1 - x
function excess_demand_good1(p1, x, σ)
    α1 = x
    α2 = 1.0 - x

    c11 = demand_good1(p1, α1, σ, ω11, ω12)   # agent 1, good 1
    c21 = demand_good1(p1, α2, σ, ω21, ω22)   # agent 2, good 1

    return c11 + c21 - ω1_total               # demand - supply
end

#find equilibrium price p1 for given (x, σ) using bisection
function find_equilibrium_price(x, σ;
        p_low = 0.01, p_high = 20.0,
        tol = 1e-8, max_iter = 1000)

    f_low  = excess_demand_good1(p_low,  x, σ)
    f_high = excess_demand_good1(p_high, x, σ)

    # we want a bracket with opposite signs
    if f_low * f_high > 0
        error("Bisection interval does not bracket a root for x = $x, σ = $σ")
    end

    for iter in 1:max_iter
        p_mid = 0.5 * (p_low + p_high)
        f_mid = excess_demand_good1(p_mid, x, σ)

        # stopping criterion: either function close to zero
        # or interval sufficiently small
        if abs(f_mid) < tol || (p_high - p_low) / 2 < tol
            return p_mid
        end

        if f_low * f_mid < 0
            p_high = p_mid
            f_high = f_mid
        else
            p_low  = p_mid
            f_low  = f_mid
        end
    end

    error("Bisection did not converge for x = $x, σ = $σ")
end

# helper: equilibrium bundles for both agents, given (p1, x, σ)
function equilibrium_bundles(p1, x, σ)
    α1 = x
    α2 = 1.0 - x

    c11 = demand_good1(p1, α1, σ, ω11, ω12)
    c12 = demand_good2(p1, α1, σ, ω11, ω12)

    c21 = demand_good1(p1, α2, σ, ω21, ω22)
    c22 = demand_good2(p1, α2, σ, ω21, ω22)

    return c11, c12, c21, c22
end

#grid of x = (0, 1)
xs = collect(range(0.01, 0.99; length = 100))

#task 1 and task 3:
# equilibrium price p1(x) for σ = 0.2 and σ = 5

p1_low  = Float64[]
p1_high = Float64[]

# quick numerical equilibrium check on a few points
function check_markets(x, σ)
    p1 = find_equilibrium_price(x, σ)
    c11, c12, c21, c22 = equilibrium_bundles(p1, x, σ)
    ex1 = (c11 + c21) - ω1_total
    ex2 = (c12 + c22) - ω2_total
    println("x = ", x, ", σ = ", σ,
            " → excess good 1 = ", ex1,
            ", excess good 2 = ", ex2)
end

check_markets(0.2, sigma_low)
check_markets(0.5, sigma_low)
check_markets(0.8, sigma_low)
check_markets(0.2, sigma_high)
check_markets(0.5, sigma_high)
check_markets(0.8, sigma_high)


# we will also store c1,1(x) and c2,1(x) here (for tasks 2 and 3)
c11_low  = Float64[]
c21_low  = Float64[]
c11_high = Float64[]
c21_high = Float64[]

for x in xs
    # σ = 0.2
    p1_star_low = find_equilibrium_price(x, sigma_low)
    push!(p1_low, p1_star_low)
    c11, _, c21, _ = equilibrium_bundles(p1_star_low, x, sigma_low)
    push!(c11_low, c11)
    push!(c21_low, c21)

    # σ = 5.0
    p1_star_high = find_equilibrium_price(x, sigma_high)
    push!(p1_high, p1_star_high)
    c11h, _, c21h, _ = equilibrium_bundles(p1_star_high, x, sigma_high)
    push!(c11_high, c11h)
    push!(c21_high, c21h)
end

#plot equilibrium p1 as a function of x (both sigmas on one plot)
plot_p1 = plot(xs, p1_low,
               label = "σ = 0.2",
               xlabel = "x (α₁)",
               ylabel = "equilibrium p₁",
               title  = "Equilibrium price p₁(x)")
plot!(xs, p1_high, label = "σ = 5.0")
display(plot_p1)

# task 2: c1,1(x) and c2,1(x) for σ = 0.2
plot_c1_low = plot(xs, c11_low,
                   label = "agent 1, good 1, σ = 0.2",
                   xlabel = "x (α₁)",
                   ylabel = "consumption of good 1",
                   title  = "Equilibrium allocations of good 1 (σ = 0.2)")
plot!(xs, c21_low, label = "agent 2, good 1, σ = 0.2")
display(plot_c1_low)

# task 3: c1,1(x) and c2,1(x) for σ = 5
plot_c1_high = plot(xs, c11_high,
                    label = "agent 1, good 1, σ = 5.0",
                    xlabel = "x (α₁)",
                    ylabel = "consumption of good 1",
                    title  = "Equilibrium allocations of good 1 (σ = 5.0)")
plot!(xs, c21_high, label = "agent 2, good 1, σ = 5.0")
display(plot_c1_high)

# task 5: values of x where the two agents consume (approximately) equal amounts of both goods
tol_equal = 1e-4

function find_equal_bundles(σ; tol = tol_equal)
    xs_equal = Float64[]
    for x in xs
        p1 = find_equilibrium_price(x, σ)
        c11, c12, c21, c22 = equilibrium_bundles(p1, x, σ)
        if abs(c11 - c21) < tol && abs(c12 - c22) < tol
            push!(xs_equal, x)
        end
    end
    return xs_equal
end

equal_x_low  = find_equal_bundles(sigma_low)
equal_x_high = find_equal_bundles(sigma_high)

println("# task 5: x where both agents have (approximately) the same bundle")
println("σ = ", sigma_low,  " → x ≈ ", equal_x_low)
println("σ = ", sigma_high, " → x ≈ ", equal_x_high)

#=
comments

task 4 – prices and allocations
- σ = 0.2 (strong complements): p1(x) changes a lot when x changes. Price is the main
  adjustment channel and reacts strongly.
- σ = 5.0 (close substitutes): p1(x) barely moves with x. Quantities adjust instead.
- Allocations react more strongly when σ = 5.0 (high substitutability), not when σ = 0.2.

task 5 – equal bundles
- On x = (0.01, 0.99) there is no value where both agents choose exactly the same bundle.
  The closest case is around x ≈ 0.5 for σ = 5.

extra question – short summary
- Higher σ shifts adjustment from prices → quantities.
- Low σ: prices very sensitive to x, quantities move little.
- High σ: prices barely react, quantities adjust a lot.
=#
