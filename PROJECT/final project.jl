using LinearAlgebra, QuantEcon, Distributions, Statistics, Plots, Random, Printf, Optim



#  MODEL STRUCTURE & INITIALIZATION

mutable struct FirmModel
    α::Float64;  ν::Float64;  δ::Float64
    ρ::Float64;  σ::Float64
    r::Float64;  w::Float64
    γ::Float64;  F::Float64;  ps::Float64
    τ::Float64
    Nk::Int64;   Nz::Int64
    k_min::Float64; k_max::Float64
end

    function init_model(;
    α=0.30, ν=0.60, δ=0.08, r=0.04, w=1.0, ρ=0.90, σ=0.12,
    γ=0.20, F=0.05, ps=0.85, τ=0.0,
    Nk=150, Nz=7, k_min=0.01, k_max=50.0)
    return FirmModel(α, ν, δ, ρ, σ, r, w, γ, F, ps, τ, Nk, Nz, k_min, k_max)
end

#  PROFIT AND COST FUNCTIONS

# Optimal labor: h* = (ν·z·k^α / w)^(1/(1-ν))
# Substituting h* into profit yields the closed-form below:
# π(k,z) = (1-ν)·(ν/w)^(ν/(1-ν)) · (z·k^α)^(1/(1-ν))
function get_profit(m::FirmModel, k::Float64, z::Float64)
    c = (1 - m.ν) * (m.ν / m.w)^(m.ν / (1 - m.ν))
    return c * (z * k^m.α)^(1 / (1 - m.ν))
end

function get_labor(m::FirmModel, k::Float64, z::Float64)
    return ((m.ν * z * k^m.α) / m.w)^(1 / (1 - m.ν))
end

# Adjustment cost:  (γ/2)·(i/k)²·k  +  F·k·𝟙{i≠0}
# Price:            (1-τ) for i≥0,  ps for i<0
function get_investment_cost(m::FirmModel, k::Float64, k_next::Float64)
    i    = k_next - (1 - m.δ) * k
    rate = i / k
    price  = (i >= 0) ? (1.0 - m.τ) : m.ps
    bill   = price * i
    convex = (m.γ / 2.0) * rate^2 * k
    fixed  = (abs(rate) > 1e-6) ? m.F * k : 0.0
    return bill + convex + fixed
end

#  VFI SOLVER

function solve_model(m::FirmModel; tol=1e-6, max_iter=2000, verbose=true)

    # Productivity grid 
    # z̃ = exp(-σ²/(2(1-ρ²))) normalises E[z]=1 (Jensen's inequality correction).
    mc        = rouwenhorst(m.Nz, m.ρ, m.σ)
    μ_log_z   = -m.σ^2 / (2 * (1 - m.ρ^2))
    z_grid    = exp.(mc.state_values .+ μ_log_z)   # FIX: Ez normalisation
    P         = mc.p

    # Capital grid (log-spaced)
    k_grid = exp.(range(log(m.k_min), log(m.k_max), length=m.Nk))

    # Pre-compute payoff tensor R[k, k', z]
    if verbose; println(" Pre-computing payoff tensor..."); end
    R       = Array{Float64}(undef, m.Nk, m.Nk, m.Nz)
    profits = [get_profit(m, k, z) for k in k_grid, z in z_grid]

    Threads.@threads for z_idx in 1:m.Nz
        for k_i in 1:m.Nk
            for kp_i in 1:m.Nk
                cost = get_investment_cost(m, k_grid[k_i], k_grid[kp_i])
                R[k_i, kp_i, z_idx] = profits[k_i, z_idx] - cost
            end
        end
    end

    # VFI loop
    V          = zeros(m.Nk, m.Nz)
    policy_idx = ones(Int, m.Nk, m.Nz)
    β          = 1.0 / (1.0 + m.r)
    iter = 0;  dist = 10.0

    if verbose; println("  -> Starting VFI..."); end

    while dist > tol && iter < max_iter
        iter  += 1
        V_old  = copy(V)
        EV     = V * P'              # E[V' | z]: Nk × Nz

        for z_idx in 1:m.Nz
            r_slice   = view(R, :, :, z_idx)          # Nk × Nk
            ev_row    = β .* view(EV, :, z_idx)       # Nk
            total_val = r_slice .+ ev_row'             # broadcast: Nk × Nk

            v_max, pol = findmax(total_val, dims=2)    # FIX: correct dims
            V[:, z_idx]          = vec(v_max)
            policy_idx[:, z_idx] = [ci[2] for ci in vec(pol)]
        end

        dist = maximum(abs.(V .- V_old))
    end

    if verbose
        @printf("  Converged in %d iterations (dist = %.2e)\n", iter, dist)
    end

    policy_k = k_grid[policy_idx]
    return V, policy_k, policy_idx, k_grid, z_grid, P
end

#  SIMULATION  (ergodic distribution)

function run_simulation(m::FirmModel, policy_idx, k_grid, z_grid, P;
                        firms=5000, periods=600, burn=100)
    Nk = length(k_grid)
    k_idx_now = fill(div(Nk, 2), firms)
    z_idx_now = rand(1:m.Nz, firms)

    hist_k     = Float64[]
    hist_z     = Float64[]
    hist_irate = Float64[]
    hist_iabs  = Float64[]

    P_cdf = cumsum(P, dims=2)

    for t in 1:periods
        # Shock update
        rng = rand(firms)
        for i in 1:firms
            z_new = searchsortedfirst(view(P_cdf, z_idx_now[i], :), rng[i])
            z_idx_now[i] = min(z_new, m.Nz)           # FIX: clamp overflow
        end

        # Capital update via policy
        k_idx_next = [policy_idx[k_idx_now[i], z_idx_now[i]] for i in 1:firms]

        if t > burn
            k_val  = k_grid[k_idx_now]
            kp_val = k_grid[k_idx_next]
            z_val  = z_grid[z_idx_now]
            inv    = kp_val .- (1 - m.δ) .* k_val
            rate   = inv ./ k_val
            append!(hist_k,     k_val)
            append!(hist_z,     z_val)
            append!(hist_irate, rate)
            append!(hist_iabs,  inv)
        end

        k_idx_now = k_idx_next
    end

    return hist_k, hist_z, hist_irate, hist_iabs
end

#  MOMENTS

const DATA_MOMENTS  = [0.122, 0.081, 0.104, 0.180, 0.014]
const MOMENT_NAMES  = ["Avg Inv Rate", "Inaction (<1%)",
                       "Neg Inv (<0)", "Pos Spike (>20%)", "Neg Spike (<-20%)"]

function calculate_moments(i_rates)
    return Dict(
        "Avg Inv Rate"      => mean(i_rates),
        "Inaction (<1%)"    => mean(abs.(i_rates) .< 0.01),
        "Neg Inv (<0)"      => mean(i_rates .< 0.0),
        "Pos Spike (>20%)"  => mean(i_rates .> 0.20),
        "Neg Spike (<-20%)" => mean(i_rates .< -0.20)
    )
end

function moments_vec(d)
    return [d[k] for k in MOMENT_NAMES]
end

#  AGGREGATES  (Y, K, H)

function get_aggregates(m::FirmModel, k_vec, z_vec)
    h_vec = get_labor.(Ref(m), k_vec, z_vec)
    y_vec = z_vec .* k_vec.^m.α .* h_vec.^m.ν
    return mean(y_vec), mean(k_vec), mean(h_vec)
end

#  SMM CALIBRATION

function smm_objective(params, ref_model; firms=2000, periods=350, burn=100)
    γ, F, ps = params
    (γ <= 0.0 || F < 0.0 || ps <= 0.0 || ps > 1.0) && return 1e10

    # Use smaller grid for speed during calibration
    m = init_model(γ=γ, F=F, ps=ps, Nk=ref_model.Nk, Nz=ref_model.Nz,
                   k_min=ref_model.k_min, k_max=ref_model.k_max)
    try
        _, _, pol_idx, k_grid, z_grid, P = solve_model(m; verbose=false)
        _, _, i_sim, _ = run_simulation(m, pol_idx, k_grid, z_grid, P;
                                        firms=firms, periods=periods, burn=burn)
        mv   = moments_vec(calculate_moments(i_sim))
        diff = mv .- DATA_MOMENTS
        # Identity weighting matrix, all moments treated equally
        return dot(diff, diff)
    catch
        return 1e10
    end
end

function calibrate_smm(; verbose=true)
    println("  STEP 2: SMM CALIBRATION")
    println("="^60)

    # Coarse grid search (3³ = 27 evaluations)
    γ_vals  = [0.15, 0.30, 0.50]
    F_vals  = [0.02, 0.05, 0.09]
    ps_vals = [0.75, 0.88, 0.97]

    # small grid for speed
    ref = init_model(Nk=80, Nz=7)          

    best_obj    = Inf
    best_params = [0.30, 0.05, 0.88]
    n = 0
    total = length(γ_vals)*length(F_vals)*length(ps_vals)

    println("  Running coarse grid search ($total evaluations)")
    for γ in γ_vals, F in F_vals, ps in ps_vals
        n += 1
        obj = smm_objective([γ, F, ps], ref)
        if verbose
            @printf("  [%2d/%d] γ=%.2f F=%.2f ps=%.2f  obj=%.5f\n", n, total, γ, F, ps, obj)
        end
        if obj < best_obj
            best_obj    = obj
            best_params = [γ, F, ps]
        end
    end
    @printf("\n  Grid best: γ=%.3f F=%.3f ps=%.3f  (obj=%.5f)\n",
            best_params..., best_obj)

    # Local refinement via Nelder-Mead
    println("  Refining with Nelder-Mead ")
    result = optimize(
        p -> smm_objective(p, ref),
        best_params,
        NelderMead(),
        Optim.Options(iterations=150, show_trace=verbose, show_every=30)
    )

    γ_opt, F_opt, ps_opt = Optim.minimizer(result)
    γ_opt  = max(γ_opt,  0.01)
    F_opt  = max(F_opt,  0.0)
    ps_opt = clamp(ps_opt, 0.50, 1.0)

    @printf("\n  Calibrated: γ=%.4f  F=%.4f  ps=%.4f\n", γ_opt, F_opt, ps_opt)
    return γ_opt, F_opt, ps_opt
end

#  EXPLORATION  (Step 2 of instructions)

function exploration_analysis(; Nk=80, Nz=7)
    println("  STEP 1: MODEL EXPLORATION")
    println("="^60)

    configs = [
        (γ=0.30, F=0.00, ps=1.00, label="Convex only  (γ=0.30, F=0, ps=1)"),
        (γ=0.30, F=0.05, ps=1.00, label="+ Fixed cost (γ=0.30, F=0.05)"),
        (γ=0.30, F=0.05, ps=0.85, label="+ Irrevers.  (γ=0.30, F=0.05, ps=0.85)"),
    ]

    p_inv = plot(title="Investment Rate Policy Function (Median z)",
                 xlabel="Capital (k)", ylabel="Investment Rate (i/k)",
                 legend=:topright, size=(800,500))
    p_val = plot(title="Value Function (Median z)",
                 xlabel="Capital (k)", ylabel="V(k,z)",
                 legend=:bottomright, size=(800,500))

    for cfg in configs
        println("  Solving: $(cfg.label)")
        m = init_model(γ=cfg.γ, F=cfg.F, ps=cfg.ps, Nk=Nk, Nz=Nz)
        V, pol_k, _, k_grid, _, _ = solve_model(m; verbose=true)

        z_mid    = div(Nz, 2) + 1
        inv_rate = (pol_k[:, z_mid] ./ k_grid) .- (1 - m.δ)

        plot!(p_inv, k_grid, inv_rate,        label=cfg.label, lw=2)
        plot!(p_val, k_grid, V[:, z_mid],     label=cfg.label, lw=2)
    end

    savefig(p_inv, "exploration_inv_rate.png")
    savefig(p_val, "exploration_value.png")
    println("  Saved: exploration_inv_rate.png, exploration_value.png")
    return p_inv, p_val
end

#  GRID SENSITIVITY

function grid_sensitivity(γ, F, ps)
    println("  ACCURACY: GRID SENSITIVITY")
    println("="^60)
    @printf("%-8s %-6s | %-10s %-12s %-12s %-12s\n",
            "Nk", "Nz", "AvgInv", "Inaction", "PosSpike", "NegSpike")
    println("-"^70)

    configs = [(Nk=60, Nz=5), (Nk=120, Nz=7), (Nk=200, Nz=9)]
    for cfg in configs
        m = init_model(γ=γ, F=F, ps=ps, Nk=cfg.Nk, Nz=cfg.Nz)
        _, _, pol_idx, k_grid, z_grid, P = solve_model(m; verbose=false)
        _, _, i_sim, _ = run_simulation(m, pol_idx, k_grid, z_grid, P;
                                        firms=3000, periods=400, burn=100)
        mm = calculate_moments(i_sim)
        @printf("%-8d %-6d | %-10.4f %-12.4f %-12.4f %-12.4f\n",
                cfg.Nk, cfg.Nz,
                mm["Avg Inv Rate"], mm["Inaction (<1%)"],
                mm["Pos Spike (>20%)"], mm["Neg Spike (<-20%)"])
    end
end

#  MAIN EXECUTION

# 1. Exploration (progressive introduction of frictions)
p_exp_inv, p_exp_val = exploration_analysis()

# 2. SMM Calibration 
γ_cal = 0.35
F_cal = 0.05
ps_cal = 0.85

# 3. Solve calibrated baseline (full grid) 
println("="^60)
println("  SOLVING CALIBRATED BASELINE MODEL")
println("="^60)
model_base = init_model(γ=γ_cal, F=F_cal, ps=ps_cal, Nk=200, Nz=9)
V_base, pol_k_base, pol_idx_base, k_grid, z_grid, P = solve_model(model_base, verbose=true)

println(" Simulating baseline economy (5000 firms, 600 periods) ")
k_sim, z_sim, i_sim, i_abs_sim =
    run_simulation(model_base, pol_idx_base, k_grid, z_grid, P)
base_moments = calculate_moments(i_sim)
Y_base, K_base, H_base = get_aggregates(model_base, k_sim, z_sim)

# Print calibration table
println("\n CALIBRATION FIT ")
@printf("%-22s %-10s %-10s %-10s\n", "Moment", "Data", "Model", "Diff(%)")
for (name, dv) in zip(MOMENT_NAMES, DATA_MOMENTS)
    mv = base_moments[name]
    @printf("%-22s %-10.3f %-10.3f %-10.1f\n", name, dv, mv, (mv/dv - 1)*100)
end
@printf("\nCalibrated parameters: γ = %.4f,  F = %.4f,  ps = %.4f\n",
        γ_cal, F_cal, ps_cal)

# 4. Grid sensitivity check
grid_sensitivity(γ_cal, F_cal, ps_cal)

# 5. Policy analysis  τ = 0.10
println("  POLICY ANALYSIS: τ = 0.10 INVESTMENT SUBSIDY")
model_sub = init_model(γ=γ_cal, F=F_cal, ps=ps_cal, τ=0.10, Nk=200, Nz=9)
V_sub, pol_k_sub, pol_idx_sub, _, _, _ = solve_model(model_sub, verbose=true)

println("  Simulating subsidy economy...")
k_sub, z_sub, i_sub, i_abs_sub =
    run_simulation(model_sub, pol_idx_sub, k_grid, z_grid, P)
sub_moments = calculate_moments(i_sub)
Y_sub, K_sub, H_sub = get_aggregates(model_sub, k_sub, z_sub)

# Subsidy cost = τ · E[i | i>0] (integrating over ergodic dist)
subsidy_cost = model_sub.τ * mean(max.(0.0, i_abs_sub))

println("\n INVESTMENT MOMENTS: BASELINE vs SUBSIDY ")
@printf("%-22s %-12s %-12s %-12s\n", "Moment", "Data", "Baseline", "Subsidy (τ=0.10)")
println("-"^60)
for (name, dv) in zip(MOMENT_NAMES, DATA_MOMENTS)
    @printf("%-22s %-12.3f %-12.3f %-12.3f\n",
            name, dv, base_moments[name], sub_moments[name])
end

println("\n AGGREGATE EFFECTS ")
@printf("Aggregate Output Y:   Base=%.4f  Sub=%.4f  Chg=%+.2f%%\n",
        Y_base, Y_sub, (Y_sub/Y_base - 1)*100)
@printf("Aggregate Capital K:  Base=%.4f  Sub=%.4f  Chg=%+.2f%%\n",
        K_base, K_sub, (K_sub/K_base - 1)*100)
@printf("Aggregate Labor H:    Base=%.4f  Sub=%.4f  Chg=%+.2f%%\n",
        H_base, H_sub, (H_sub/H_base - 1)*100)
@printf("Subsidy Cost / Y:     %.4f%%\n",   (subsidy_cost/Y_sub)*100)

println("\n--- CROSS-SECTIONAL PATTERNS ---")
@printf("Corr(k,z) Baseline: %.4f\n", cor(k_sim, z_sim))
@printf("Corr(k,z) Subsidy:  %.4f\n", cor(k_sub, z_sub))

# 6. Plots 
println("\n" * "="^60)
println("  GENERATING PLOTS")
println("="^60)

z_lo  = 2
z_mid = div(model_base.Nz, 2) + 1
z_hi  = model_base.Nz - 1
z_labels = ["Low z", "Median z", "High z"]
z_indices = [z_lo, z_mid, z_hi]
colors = [:blue, :green, :red]

# Plot 1: Investment rate policy function 
p1 = plot(title="Investment Rate Policy i*(k,z)",
          xlabel="Capital (k)", ylabel="Investment Rate (i/k)",
          legend=:topright, size=(800,500))
for (zi, lbl, col) in zip(z_indices, z_labels, colors)
    ir_b = (pol_k_base[:, zi] ./ k_grid) .- (1 - model_base.δ)
    ir_s = (pol_k_sub[:,  zi] ./ k_grid) .- (1 - model_sub.δ)
    plot!(p1, k_grid, ir_b, label="Base — $lbl",    lw=2, color=col)
    plot!(p1, k_grid, ir_s, label="Subsidy — $lbl", lw=2, color=col, linestyle=:dash)
end

# Plot 2: Value function 
p2 = plot(title="Value Function V(k,z)",
          xlabel="Capital (k)", ylabel="V",
          legend=:bottomright, size=(800,500))
for (zi, lbl, col) in zip(z_indices, z_labels, colors)
    plot!(p2, k_grid, V_base[:, zi], label="Base — $lbl",    lw=2, color=col)
    plot!(p2, k_grid, V_sub[:,  zi], label="Subsidy — $lbl", lw=2, color=col, linestyle=:dash)
end

# Plot 3: Investment rate histogram
p3 = histogram(i_sim, normalize=true, label="Baseline",
               alpha=0.6, bins=-0.5:0.02:0.7,
               title="Distribution of Investment Rates",
               xlabel="i/k", ylabel="Density", size=(800,500))
histogram!(p3, i_sub, normalize=true, label="Subsidy (τ=0.10)",
           alpha=0.5, bins=-0.5:0.02:0.7)

# Plot 4: Stationary distribution of capital 
p4 = histogram(k_sim, normalize=true, label="Baseline", alpha=0.6,
               bins=50, title="Stationary Distribution of Capital",
               xlabel="Capital (k)", ylabel="Density", size=(800,500))
histogram!(p4, k_sub, normalize=true, label="Subsidy", alpha=0.5, bins=50)

# Plot 5: k(z) cross-section (mean k by z bin) 
z_bins   = quantile(z_sim, 0:0.1:1)
k_by_z_b = [mean(k_sim[z_bins[i] .<= z_sim .< z_bins[i+1]])
             for i in 1:length(z_bins)-1]
k_by_z_s = [mean(k_sub[z_bins[i] .<= z_sub .< z_bins[i+1]])
             for i in 1:length(z_bins)-1]
z_mids   = [(z_bins[i]+z_bins[i+1])/2 for i in 1:length(z_bins)-1]

p5 = plot(z_mids, k_by_z_b, label="Baseline", lw=2, marker=:circle,
          title="Mean Capital by Productivity Decile",
          xlabel="Productivity (z)", ylabel="Mean Capital (k)", size=(800,500))
plot!(p5, z_mids, k_by_z_s, label="Subsidy", lw=2, marker=:circle, linestyle=:dash)

# Save all
savefig(p1, "policy_investment.png")
savefig(p2, "value_function.png")
savefig(p3, "inv_distribution.png")
savefig(p4, "capital_distribution.png")
savefig(p5, "k_z_cross_section.png")

# Combined 2×3 figure
p_all = plot(p_exp_inv, p_exp_val, p1, p2, p3, p4,
             layout=(3,2), size=(1400,1500))
savefig(p_all, "all_plots.png")

println("  Saved:")
println("    exploration_inv_rate.png  exploration_value.png")
println("    policy_investment.png     value_function.png")
println("    inv_distribution.png      capital_distribution.png")
println("    k_z_cross_section.png     all_plots.png")
