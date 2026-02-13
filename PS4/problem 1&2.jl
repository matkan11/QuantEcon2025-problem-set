using QuantEcon, Interpolations, Optim, Plots, Statistics, Random, LaTeXStrings, Printf

#Grid setup

struct Model
    γ::Float64          
    β::Float64          
    R::Float64          
    ρ::Float64          
    σ_eps::Float64      

    #grids
    z_grid::Vector{Float64}
    Π::Matrix{Float64} 
    a_grid::Vector{Float64}
    a_min::Float64

    method_name::String
end

function setup_model(; γ=2.0, method="Standard")
    β = 0.99
    ρ = 0.90
    σ_eps = 0.20 * sqrt(1 - ρ^2)
    #interest basing on gamma
    R = (γ == 2.0) ? 1.010 : 1.008

    #income discretization
    mc = rouwenhorst(11, ρ, σ_eps)
    z_grid = exp.(mc.state_values)
    Π = mc.p

    #asset grid
    z_min = minimum(z_grid)
    a_min = -0.6 * (z_min / (R - 1)) 
    a_max = 500.0
    N_a = 100

    ω = range(0, 1, length=N_a)
    θ = 3.0
    a_grid = a_min .+ (a_max - a_min) .* (ω.^θ)

    return Model(γ, β, R, ρ, σ_eps, z_grid, Π, a_grid, a_min, method)
end

#Solving 

#utility function
u(c, γ) = (γ == 1.0) ? log(c) : (c^(1-γ))/(1-γ)

#solver function

function solve_model(m::Model; tol=1e-7, max_iter=2000)
    Nz = length(m.z_grid)
    Na = length(m.a_grid)

    #value function. for transformed guess proportional to wealth, for standard u(wealth)
    if m.method_name == "Transformed"
        wealth = repeat(m.a_grid, 1, Nz) .+ reshape(m.z_grid, 1, Nz)
        V = max.(wealth, 1e-5)
    else 
        wealth = repeat(m.a_grid, 1, Nz) .+ reshape(m.z_grid, 1, Nz)
        V = u.(max.(wealth, 1e-3), m.γ)
    end

    V_new = similar(V)
    policy_a = similar(V)
    policy_c = similar(V)

    iter = 0
    dist = 1.0

    t_start = time()

    while dist > tol && iter < max_iter
        iter += 1

        #expected value of interpolation for different methods
        if m.method_name == "Transformed"
            V_power = V.^(1 - m.γ)
            EV_raw = V_power * m.Π'
        else
            EV_raw = V * m.Π'
        end

        #creating interpolation objects for each state z and interpolate over a'
        itps = [linear_interpolation(m.a_grid, EV_raw[:, iz], extrapolation_bc=Line()) for iz in 1:Nz]

        for iz in 1:Nz
            z = m.z_grid[iz]
            itp = itps[iz]

            for ia in 1:Na
                a = m.a_grid[ia]

                #max feasible a' is for c=0, a = Ra + z, min is borrowing limit
                bound_hi = m.R * a + z - 1e-8 #small number to avoid errors with 0
                bound_lo = m.a_min
                
                if bound_hi < bound_lo
                    bound_hi = bound_lo
                end

                #objective function
                function objective(ap)
                    c = m.R * a + z - ap
                    if c <= 1e-10 
                        return 1e20
                    end

                    if m.method_name == "Transformed"
                        #to maximize, need to verify direction because if γ > 1 we need to minimize
                        
                        ev_val = itp(ap) 
                        inner = (1 - m.β) * c^(1 - m.γ) + m.β * ev_val

                        val = inner^(1/(1-m.γ))
                        return -val
                    else
                        ev_val = itp(ap)
                        val = u(c, m.γ) + m.β * ev_val
                        return -val
                    end
                end

                #optimizing using Brent's method 
                res = optimize(objective, bound_lo, bound_hi, Brent())

                policy_a[ia, iz] = Optim.minimizer(res)
                policy_c[ia, iz] = m.R * a + z - policy_a[ia, iz]
                V_new[ia, iz] = -Optim.minimum(res)
            end
        end

        #check for convergence
        diff = abs.(V_new .- V) ./ (1 .+ abs.(V))
        dist = maximum(diff)

        V .= V_new
    end

    runtime = time() - t_start
    println("Method: $(m.method_name) | γ=$(m.γ) | Iter: $iter | Time: $(round(runtime, digits=2))s")

    return V, policy_a, policy_c, runtime
end

#Euler errors and simulation 

function compute_euler_errors(m::Model, policy_c)

    Na, Nz = size(policy_c)
    ee_errors = zeros(Na, Nz)

    #we need to interpolate c(a', z') to evaluate E[...] and since a is continuous, we interpolate policy c for next period
    c_itps = [linear_interpolation(m.a_grid, policy_c[:, iz], extrapolation_bc=Line()) for iz in 1:Nz]

    for iz in 1:Nz
        for ia in 1:Na
            c_curr = policy_c[ia, iz]
            a_prime = m.R * m.a_grid[ia] + m.z_grid[iz] - c_curr

            #expectation 
            E_mu_next = 0.0
            for iz_next in 1:Nz
                #find c and clamping a for safety
                a_p_clamped = clamp(a_prime, m.a_min, maximum(m.a_grid))
                c_next = c_itps[iz_next](a_p_clamped)
                mu_next = c_next^(-m.γ)
                E_mu_next += m.Π[iz, iz_next] * mu_next
            end

            rhs = m.β * m.R * E_mu_next
            c_implied = rhs^(-1/m.γ)

            ee_errors[ia,iz] = log10(abs(1.0 - c_implied / c_curr) + 1e-16)
        end
    end
    return ee_errors
end


function simulate_series(m::Model, policy_a, policy_c, T=10000, burn=500)
    Random.seed!(1234)

    Nz = length(m.z_grid)

    #simulating income chain
    z_indices = simulate_indices(mc_from_model(m), T + burn, init=6) # 6 is median of 11
    z_sim = m.z_grid[z_indices]
    
    #simulating assets
    a_sim = zeros(T + burn)
    c_sim = zeros(T + burn)
    ee_sim = zeros(T + burn)

    a_sim[1] = 0.0

    #interpolators for policy functions 
    a_itps = [linear_interpolation(m.a_grid, policy_a[:, iz], extrapolation_bc=Line()) for iz in 1:Nz]
    c_itps = [linear_interpolation(m.a_grid, policy_c[:, iz], extrapolation_bc=Line()) for iz in 1:Nz]


    for t in 1:(T + burn - 1)
        iz = z_indices[t]
        a_curr = a_sim[t]

        #evaluating policies, bounding for safety
        a_curr_clamped = clamp(a_curr, m.a_min, maximum(m.a_grid))

        #decision
        ap = a_itps[iz](a_curr_clamped)
        c = c_itps[iz](a_curr_clamped)

        a_sim[t+1] = ap
        c_sim[t] = c
    end
    #filling last c
    c_sim[end] = c_itps[z_indices[end]](a_sim[end])

    #discarding burn-ins 
    return a_sim[burn+1:end], c_sim[burn+1:end], z_sim[burn+1:end]
end

#helper function for MC for simulation
function mc_from_model(m::Model)
    return MarkovChain(m.Π, m.z_grid)
end


#execution

function run_problem1()
    results = Dict()

    for γ in [2.0, 10.0]
        println("\n--- Solving for Gamma = $γ ---")

        #solving standard
        m_std = setup_model(γ=γ, method="Standard")
        V_std, pol_a_std, pol_c_std, time_std = solve_model(m_std)
        ee_std = compute_euler_errors(m_std, pol_c_std)

        #solving transformed
        m_trans = setup_model(γ=γ, method="Transformed")
        V_trans, pol_a_trans, pol_c_trans, time_trans = solve_model(m_trans)
        ee_trans = compute_euler_errors(m_trans, pol_c_trans)

        #simulations
        sim_a_s, sim_c_s, sim_z = simulate_series(m_std, pol_a_std, pol_c_std)
        sim_a_t, sim_c_t, _     = simulate_series(m_trans, pol_a_trans, pol_c_trans)

        #reporting of statistics 
        frac_std = mean(sim_a_s[2:end] .<= m_std.a_min + 1e-3)
        frac_trans = mean(sim_a_t[2:end] .<= m_trans.a_min + 1e-3)
        
        println("\nResults for Gamma $γ:")
        println("Standard Method Time: $time_std")
        println("Transform Method Time: $time_trans")
        println("Mean Assets (Std vs Trans): $(mean(sim_a_s)) vs $(mean(sim_a_t))")
        println("Frac at Constraint (Std vs Trans): $frac_std vs $frac_trans")

        #plotting storing
        results[γ] = (m_std, pol_a_std, pol_c_std, ee_std, 
                      m_trans, pol_a_trans, pol_c_trans, ee_trans,
                      sim_a_s, sim_c_s, sim_a_t, sim_c_t, sim_z)
    end
    return results
end


results = run_problem1()





#plotting

γ = 2.0
data = results[γ]
m_std, pa_s, pc_s, ee_s, m_t, pa_t, pc_t, ee_t, sa_s, sc_s, sa_t, sc_t, sz = data

#Plot 1  Policy Functions
# Income indices: Low (1), Median (6), High (11)
indices = [1, 6, 11]
labels = ["Low Income", "Median Income", "High Income"]

p1 = plot(title="Consumption Policy (Transformed, γ=2)", xlabel="Assets", ylabel="Consumption")
for (i, idx) in enumerate(indices)
    plot!(p1, m_t.a_grid, pc_t[:, idx], label=labels[i], lw=2)
end

p2 = plot(title="Savings Policy (Transformed, γ=2)", xlabel="Assets", ylabel="Assets Next (a')", legend=:bottomright)
plot!(p2, m_t.a_grid, m_t.a_grid, label="45 degree", linestyle=:dash, color=:black)
for (i, idx) in enumerate(indices)
    plot!(p2, m_t.a_grid, pa_t[:, idx], label=labels[i], lw=2)
end

#Plot 3 Euler Errors
p3 = plot(title="Euler Errors (Standard vs Transformed)", xlabel="Assets", ylabel="Log10 Error")
plot!(p3, m_std.a_grid, ee_s[:, 6], label="Standard (Median z)", lw=2)
plot!(p3, m_t.a_grid, ee_t[:, 6], label="Transformed (Median z)", lw=2)
xlims!(p3, m_t.a_min, -m_t.a_min + 2.0) #zoom


#Plot 4 Simulation Overlay
p4 = plot(title="Asset Simulation (First 100 periods)", xlabel="Time", ylabel="Assets")
plot!(p4, sa_s[1:100], label="Standard", lw=1.5)
plot!(p4, sa_t[1:100], label="Transformed", lw=1.5, linestyle=:dash)


display(plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800)))



#iterate over both gammas
for γ in [2.0, 10.0]
    (m_std, pa_s, pc_s, ee_s, 
     m_t, pa_t, pc_t, ee_t, 
     sa_s, sc_s, 
     sa_t, sc_t, sz) = results[γ]
    
    println("\nRESULTS FOR GAMMA = $γ")
    println("-"^60)
    
    #Table
    function get_stats(a_sim, c_sim, z_sim, a_min)
        μ_a, σ_a = mean(a_sim), std(a_sim)
        μ_c, σ_c = mean(c_sim), std(c_sim)
        min_a, max_a = minimum(a_sim), maximum(a_sim)
        min_c, max_c = minimum(c_sim), maximum(c_sim)
        
        #fraction at constraint
        frac = mean(a_sim[2:end] .<= a_min + 1e-3)
        
        #correlations
        ac_a = cor(a_sim[1:end-1], a_sim[2:end])
        ac_c = cor(c_sim[1:end-1], c_sim[2:end])
        corr_az = cor(a_sim, z_sim)
        corr_cz = cor(c_sim, z_sim)
        
        return [μ_a, σ_a, μ_c, σ_c, min_a, max_a, min_c, max_c, frac, ac_a, ac_c, corr_az, corr_cz]
    end

    #calculate stats
    stats_s = get_stats(sa_s, sc_s, sz, m_std.a_min)
    stats_t = get_stats(sa_t, sc_t, sz, m_t.a_min) 
    
    #pint table
    labels = ["Mean Assets", "Std Assets", "Mean Cons", "Std Cons", 
              "Min Assets", "Max Assets", "Min Cons", "Max Cons", 
              "Frac Constrained", "Autocorr(a)", "Autocorr(c)", "Corr(a,z)", "Corr(c,z)"]
    
    @printf("%-20s | %-12s | %-12s\n", "Statistic", "Standard", "Transformed")
    println("-"^50)
    for i in 1:length(labels)
        @printf("%-20s | %-12.4f | %-12.4f\n", labels[i], stats_s[i], stats_t[i])
    end

    # euler error stats
    ee_flat_t = vec(ee_t)
    p10, p50, p90 = quantile(ee_flat_t, [0.1, 0.5, 0.9])
    mean_ee = mean(ee_flat_t)
    
    println("\nEuler Error Stats (Transformed Method):")
    @printf("Mean: %.4f | 10th: %.4f | Median: %.4f | 90th: %.4f\n", mean_ee, p10, p50, p90)

    # plots
    zoom_range = 1:20 
    
    p_pol_zoom = plot(title="Savings Policy (Zoomed, γ=$γ)", xlabel="a", ylabel="a'", legend=:topleft)
    plot!(p_pol_zoom, m_t.a_grid[zoom_range], m_t.a_grid[zoom_range], linestyle=:dash, color=:black, label="45 deg")
    for i in [1, 6, 11]
        plot!(p_pol_zoom, m_t.a_grid[zoom_range], pa_t[zoom_range, i], label="z index $(i)", lw=2)
    end
    
    # simulation paths (First 100 periods)
    t_range = 1:100
    
    # Overlay Assets
    p_sim_a = plot(t_range, sa_t[t_range], title="Assets Path", label="Transformed", ylabel="a_t", lw=2)
    plot!(p_sim_a, t_range, sa_s[t_range], label="Standard", linestyle=:dash, lw=2)
    
    # Overlay Consumption
    p_sim_c = plot(t_range, sc_t[t_range], title="Consumption Path", label="Transformed", color=:orange, ylabel="c_t", lw=2)
    plot!(p_sim_c, t_range, sc_s[t_range], label="Standard", linestyle=:dash, color=:red, lw=2)
    
    # Income Process
    p_sim_z = plot(t_range, sz[t_range], title="Income Process", label="Income", color=:green, ylabel="z_t", lw=2)
    
    # Calculate and Plot Euler Errors over Time
    ee_path_t = zeros(100)
    ee_path_s = zeros(100)
    for t in t_range
        #finding z index
        iz = findfirst(x -> abs(x - sz[t]) < 1e-5, m_t.z_grid)
        
        # Interpolate Euler errors for both methods evaluated at current asset simulation points
        itp_t = linear_interpolation(m_t.a_grid, ee_t[:, iz], extrapolation_bc=Line())
        itp_s = linear_interpolation(m_std.a_grid, ee_s[:, iz], extrapolation_bc=Line())
        
        ee_path_t[t] = itp_t(clamp(sa_t[t], m_t.a_min, maximum(m_t.a_grid)))
        ee_path_s[t] = itp_s(clamp(sa_s[t], m_std.a_min, maximum(m_std.a_grid)))
    end
    
    p_sim_ee = plot(t_range, ee_path_t, title="Euler Errors Path", label="Transformed", ylabel="Log10 Error", lw=2)
    plot!(p_sim_ee, t_range, ee_path_s, label="Standard", linestyle=:dash, lw=2)
    
    # Display all 5 plots in a nice layout
    display(plot(p_pol_zoom, p_sim_a, p_sim_c, p_sim_z, p_sim_ee, layout=(3,2), size=(1000, 1200)))
end




##########################
#PROBLEM 2
#########################

#INDIVIDUAL MPCs
function compute_individual_mpc(m::Model, policy_c, Δ_vals)
    Nz = length(m.z_grid)
    Na = length(m.a_grid)
    
    #continuous consumption function
    c_itps = [linear_interpolation(m.a_grid, policy_c[:, iz], extrapolation_bc=Line()) for iz in 1:Nz]

    mpc_results = Dict()

    for Δ in Δ_vals
        mpc_grid = zeros(Na, Nz)
        
        for iz in 1:Nz
            for ia in 1:Na
                a = m.a_grid[ia]
                
                #c(a, z)
                c_base = policy_c[ia, iz]
                
                # c(a + Δ, z) requires interpolation
                c_plus = c_itps[iz](a + Δ)
                
                mpc_grid[ia, iz] = (c_plus - c_base) / Δ
            end
        end
        mpc_results[Δ] = mpc_grid
    end
    
    return mpc_results
end

best_model_data = results[2.0]
m_best = best_model_data[5]      # m_t
policy_c_best = best_model_data[7] # pc_t
policy_a_best = best_model_data[6] # pa_t

#mpcs computing
Δ_grid = [0.01, 0.1, 0.5, 1.0, 2.0]
mpc_results = compute_individual_mpc(m_best, policy_c_best, Δ_grid)

#plotting mpcs
income_indices = [1, 6, 11]
income_labels = ["Low Income", "Median Income", "High Income"]

plts = []

for (i, iz) in enumerate(income_indices)
    p = plot(title="MPC ($(income_labels[i]))", xlabel="Assets (a)", ylabel="MPC")
    
    for Δ in [0.01, 0.5, 2.0]
        mpc_vals = mpc_results[Δ][:, iz]
        plot!(p, m_best.a_grid, mpc_vals, label="Δ=$(Δ)", lw=2)
    end
    
    #zoom near constraint
    xlims!(p, m_best.a_min, 5.0)
    push!(plts, p)
end

display(plot(plts..., layout=(1,3), size=(1200, 400)))

#MPCs are highest for low-wealth households near the borrowing constraint, reaching
# almost 1.0, as they consume almost the entire transfer immediately.
# As wealth increases, the MPC drops sharply
# because unconstrained households prefer to smooth consumption over time.


#Aggregate response
using SparseArrays

#distribution functions
function build_transition_matrix(m::Model, policy_a)
    Nz = length(m.z_grid)
    Na = length(m.a_grid)
    n_states = Na * Nz

    # sparse matrix with probability
    I_idx, J_idx, V_val = Int[], Int[], Float64[]
    for iz in 1:Nz
        for iz_next in 1:Nz
            prob_z = m.Π[iz, iz_next]

            for ia in 1:Na 
                src_idx = (iz - 1) * Na + ia 
                
                #decision
                ap = policy_a[ia, iz]
                ap = clamp(ap, m.a_min, maximum(m.a_grid))

                #finding right a 
                k = searchsortedlast(m.a_grid, ap)
                if k == Na; k = Na - 1; end

                w = (ap - m.a_grid[k]) / (m.a_grid[k+1] - m.a_grid[k])

                #destination indices
                dest_low = (iz_next - 1) * Na + k
                dest_high = (iz_next - 1) * Na + (k + 1)

                push!(I_idx, dest_low);  push!(J_idx, src_idx); push!(V_val, prob_z * (1 - w))
                push!(I_idx, dest_high); push!(J_idx, src_idx); push!(V_val, prob_z * w)
            end
        end
    end
    return sparse(I_idx, J_idx, V_val, n_states, n_states)
end


function get_stationary_distribution(Λ)
    λ = fill(1.0 / size(Λ, 1), size(Λ, 1))
    for i in 1:2000
        λ_new = Λ * λ
        if maximum(abs.(λ_new - λ)) < 1e-10
            return λ_new
        end
        λ = λ_new
    end
    println("Distribution iteration hit max limit")
    return λ
end


function run_aggregate_experiment(m::Model, policy_c, Λ_star, λ_star)
    Na, Nz = length(m.a_grid), length(m.z_grid)
    c_flat = vec(policy_c) 
    
    # steady state consumption 
    C_star = sum(c_flat .* λ_star)
    
    #transfer 
    Δ_agg = 0.05 * C_star
    println("Steady State C*: $(round(C_star, digits=4))")
    println("Transfer Size Δ: $(round(Δ_agg, digits=4))")
    
    #shifted distributions 
    λ_0 = zeros(length(λ_star))

    for iz in 1:Nz
        for ia in 1:Na
            src_idx = (iz - 1) * Na + ia
            mass = λ_star[src_idx]

            if mass > 1e-16
                #shifting off grid
                a_new = clamp(m.a_grid[ia] + Δ_agg, m.a_min, maximum(m.a_grid))
                
                #interpolating mass on grid
                k = searchsortedlast(m.a_grid, a_new)
                if k == Na; k = Na - 1; end
                w = (a_new - m.a_grid[k]) / (m.a_grid[k+1] - m.a_grid[k])
                
                dest_low = (iz - 1) * Na + k
                dest_high = (iz - 1) * Na + (k + 1)
                
                λ_0[dest_low] += mass * (1 - w)
                λ_0[dest_high] += mass * w
            end
        end
    end

    T_sim = 50
    C_path = zeros(T_sim + 1)
    
    λ_curr = λ_0
    C_path[1] = sum(c_flat .* λ_curr) # t=0 (Impact)
    
    for t in 1:T_sim
        λ_curr = Λ_star * λ_curr
        C_path[t+1] = sum(c_flat .* λ_curr)
    end
    
    return C_star, C_path, Δ_agg
end

#building transition matrix 
Λ_star = build_transition_matrix(m_best, policy_a_best)

#solving for stationary distribution
λ_star = get_stationary_distribution(Λ_star)

#running transfer simulaiton 
C_star, C_path, Δ_agg = run_aggregate_experiment(m_best, policy_c_best, Λ_star, λ_star)

#impulse response
irf = (C_path .- C_star) ./ C_star

# cumulative MPCs
horizons = [0, 1, 4, 8, 12, 20] 
cum_mpcs = Dict()
excess_C = C_path .- C_star

println("\n--- Aggregate Results ---")
for H in horizons
    if H == 0
        mpc_H = 0.0
    else
        total_spent = sum(excess_C[1:H])
        mpc_H = total_spent / Δ_agg
    end
    cum_mpcs[H] = mpc_H
    println("Cumulative MPC (H=$H periods): $(round(mpc_H, digits=4))")
end

println("Fraction spent by H=12: $(round(cum_mpcs[12] * 100, digits=2))%")

p_irf = plot(0:50, irf * 100, 
    title="Aggregate Impulse Response", 
    xlabel="Periods (t)", 
    ylabel="% Deviation of C from Steady State",
    lw=2, legend=false)

display(p_irf)