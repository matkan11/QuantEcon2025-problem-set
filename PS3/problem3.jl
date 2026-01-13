#Olaf Jarosz
using Interpolations, Plots, Printf, Parameters, LinearAlgebra, Statistics, QuantEcon, Random, Distributions

@with_kw struct ModelParams
    J::Int = 60          
    γ::Float64 = 2.0     
    γb::Float64 = 1.0    
    β::Float64 = 0.96   
    r::Float64 = 0.04167 
    θ::Float64 = 0.5     
    a_bar::Float64 = 2.0 
    y_base::Float64 = 1.0
    na::Int = 500       
end

function solve_lifecycle(p::ModelParams)
    # 1. Mortality probabilities
    π = [min(0.0005 * 1.14^j, 1.0) for j in 1:p.J]
    
    # 2. Income profile
    y = [(j <= 40 ? p.y_base * (0.8 + 0.02*j) : p.y_base * 0.3) for j in 1:p.J]
    
    # 3. Asset Grid 
    amax = 150.0 * p.y_base 
    a_grid = [amax * ((i-1)/(p.na-1))^2 for i in 1:p.na]
    
    # Utility Functions
    u(c) = c > 1e-10 ? c^(1-p.γ)/(1-p.γ) : -1e10
    ϕ(a) = p.θ * (a + p.a_bar)^(1-p.γb)/(1-p.γb)
    
    # Storage for Value and Policy Functions
    V = zeros(p.J, p.na)
    pol_a = zeros(p.J, p.na)
    pol_c = zeros(p.J, p.na)

    # 4. Backward Induction
    for j in p.J:-1:1
        V_next = (j == p.J) ? nothing : LinearInterpolation(a_grid, V[j+1, :], extrapolation_bc=Line())
        
        for (i, a) in enumerate(a_grid)
            budget = (1+p.r)*a + y[j]
            
            # Optimization over a'
            best_v = -Inf
            best_aprime = 0.0
            
            # Search over grid for a'
            for ap_val in a_grid
                if ap_val >= budget
                    break
                end
                c = budget - ap_val
                
                # Expected continuation value
                if j == p.J
                    val = u(c) + p.β * ϕ(ap_val)
                else
                    cont_val = (1-π[j]) * V_next(ap_val) + π[j] * ϕ(ap_val)
                    val = u(c) + p.β * cont_val
                end
                
                if val > best_v
                    best_v = val
                    best_aprime = ap_val
                end
            end
            
            V[j, i] = best_v
            pol_a[j, i] = best_aprime
            pol_c[j, i] = budget - best_aprime
        end
    end
    
    return (V=V, pol_a=pol_a, pol_c=pol_c, a_grid=a_grid, y=y)
end

function simulate(res, p::ModelParams)
    a_path = zeros(p.J + 1)
    c_path = zeros(p.J)
    s_path = zeros(p.J)
    a_path[1] = 0.0 
    
    for j in 1:p.J
        # Interpolate policy
        itp_c = LinearInterpolation(res.a_grid, res.pol_c[j, :])
        c_path[j] = itp_c(a_path[j])
        a_path[j+1] = (1+p.r)*a_path[j] + res.y[j] - c_path[j]
        s_path[j] = a_path[j+1] - a_path[j]
    end
    return (a=a_path[1:p.J], c=c_path, y=res.y, s=s_path)
end

# Analysis & Plotting

# 1. Base Case Analysis (y_bar = 1.0, θ = 0.5)
params = ModelParams(60, 2.0, 1.0, 0.96, 0.04167, 0.5, 2.0, 1.0, 500)
res = solve_lifecycle(params)
sim = simulate(res, params)

# Plot Policy Functions
p1 = plot(res.a_grid[1:100], [res.pol_c[j, 1:100] for j in [20, 40, 60]], 
          title="Consumption Policy", labels=["J=20" "J=40" "J=60"], xlabel="Assets")
p2 = plot(res.a_grid[1:100], [res.pol_a[j, 1:100] for j in [20, 40, 60]], 
          title="Savings Policy", labels=["J=20" "J=40" "J=60"], xlabel="Assets")
display(plot(p1, p2, layout=(2,1)))

# 2. Bequest vs No Bequest for different income levels
y_levels = [0.5, 1.0, 2.0]
results_with = []
results_without = []

for y in y_levels
    p_w = ModelParams(60, 2.0, 1.0, 0.96, 0.04167, 0.5, 2.0, y, 500)
    p_wo = ModelParams(60, 2.0, 1.0, 0.96, 0.04167, 0.0, 2.0, y, 500)
    
    push!(results_with, simulate(solve_lifecycle(p_w), p_w))
    push!(results_without, simulate(solve_lifecycle(p_wo), p_wo))
end

# Wealth Inequality Calculation
peak_w = [maximum(s.a) for s in results_with]
peak_wo = [maximum(s.a) for s in results_without]

ineq_w = peak_w[3] / peak_w[1]
ineq_wo = peak_wo[3] / peak_wo[1]

println("Wealth Inequality (High/Low Income Peak Assets):")
@printf "With Bequest: %.2f\n" ineq_w
@printf "Without Bequest: %.2f\n" ineq_wo
