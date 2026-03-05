using LinearAlgebra, Statistics, Random, Printf, Optim, Interpolations, Plots

# ============================================================
# SECTION 1: PARAMETERS & GRIDS
# ============================================================

#All model parameters.
#  α, ν     : capital and labor elasticities
#  δ        : depreciation rate
#  r        : real interest rate
#  β        : discount factor = 1/(1+r)
#  w        : wage (normalized to 1)
#  ρ, σ, z̃ : AR(1) productivity process parameters
#  γ        : quadratic adjustment cost coefficient
#  F        : fixed adjustment cost (proportional to k)
#  ps       : resale price of capital (partial irreversibility; ps ≤ 1)
#  τ        : investment subsidy rate
struct Params
    α::Float64; ν::Float64; δ::Float64
    r::Float64; β::Float64; w::Float64
    ρ::Float64; σ::Float64; z̃::Float64
    γ::Float64; F::Float64; ps::Float64
    τ::Float64
end

# Grids for capital and productivity, and transition matrix for z.
function make_params(; γ=0.5, F=0.02, ps=0.95, τ=0.0)
    ρ = 0.90; σ = 0.12
    z̃ = exp(-σ^2 / (2*(1 - ρ^2)))
    Params(0.30, 0.60, 0.08,      # α, ν, δ
           0.04, 1/1.04, 1.0,     # r, β, w
           ρ, σ, z̃,               # productivity
           γ, F, ps, τ)           # adj costs + subsidy
end

struct Grids
    k_grid::Vector{Float64}
    z_grid::Vector{Float64}
    Π_z   ::Matrix{Float64}
    nk    ::Int
    nz    ::Int
end

# Model struct to hold parameters and grids together for easy passing to functions.
function rouwenhorst(ρ::Float64, σ::Float64, n::Int)
    p = (1 + ρ) / 2

    function Π_rec(n)
        n == 2 && return [p 1-p; 1-p p]
        Π_prev = Π_rec(n - 1)
        Π_new = (p     * [Π_prev zeros(n-1,1); zeros(1,n)] +
                 (1-p) * [zeros(n-1,1) Π_prev; zeros(1,n)] +
                 (1-p) * [zeros(1,n); Π_prev zeros(n-1,1)] +
                 p     * [zeros(1,n); zeros(n-1,1) Π_prev])
        for i in 1:n; Π_new[i,:] ./= sum(Π_new[i,:]); end
        return Π_new
    end

    σ_z   = σ / sqrt(1 - ρ^2)
    step  = σ_z * sqrt(n - 1)
    z_grid = exp.(range(-step, step, length=n))
    return z_grid, Π_rec(n)
end

# Rouwenhorst method to discretize AR(1) process for z. It`s better than Tauchen for high persistence. Returns z_grid and transition matrix Π_z.
function make_grids(p::Params; nk::Int=50, nz::Int=15)
    z_grid, Π_z = rouwenhorst(p.ρ, p.σ, nz)
    k_grid = exp.(range(log(0.5), log(25.0), length=nk))
    return Grids(collect(k_grid), collect(z_grid), Π_z, nk, nz)
end

# ============================================================
# SECTION 2: ANALYTICAL SOLUTIONS & PERIOD PROFIT
# ============================================================

# Optimal labor demand h*(k,z) from static profit maximization given k and z.
@inline function h_star(z::Float64, k::Float64, p::Params)
    return (p.ν * z * k^p.α / p.w)^(1.0 / (1.0 - p.ν))
end

# Operating profit 
@inline function flow_profit(z::Float64, k::Float64, i::Float64, p::Params)
    h = h_star(z, k, p)
    y = z * k^p.α * h^p.ν
    price    = i >= 0.0 ? (1.0 - p.τ) : p.ps
    c_convex = (p.γ / 2.0) * (i / k)^2 * k
    c_fixed  = p.F * k * (abs(i / k) > 1e-8)
    return y - p.w * h - price * i - c_convex - c_fixed
end

# ============================================================
# SECTION 3: VALUE FUNCTION ITERATION
# ============================================================

# Value function iteration to solve for V(k,z) and policy function i*(k,z).
function solve_vfi(p::Params, g::Grids;
                   ni::Int=300, tol::Float64=1e-6,
                   max_iter::Int=3000, verbose::Bool=true)
    nk, nz = g.nk, g.nz
    V      = zeros(nk, nz)
    V_new  = zeros(nk, nz)
    i_star = zeros(nk, nz)
    iter   = 0
    diff   = Inf

    while diff > tol && iter < max_iter
        iter += 1

        for iz in 1:nz
            # Expected continuation value E[V(k',z')|z] — shape (nk,)
            EV     = V * g.Π_z[iz, :]
            EV_itp = LinearInterpolation(g.k_grid, EV, extrapolation_bc=Line())

            for ik in 1:nk
                k      = g.k_grid[ik]
                z      = g.z_grid[iz]
                k_keep = (1.0 - p.δ) * k

                # Feasible investment: k' ∈ [k_min, k_max], i ≥ -k
                i_min = max(g.k_grid[1] - k_keep, -k)
                i_max = g.k_grid[end] - k_keep

                best_val = -Inf
                best_i   = 0.0

                for inv in range(i_min, i_max, length=ni)
                    val = flow_profit(z, k, inv, p) + p.β * EV_itp(k_keep + inv)
                    if val > best_val
                        best_val = val
                        best_i   = inv
                    end
                end

                # Explicitly check inaction (i=0) when fixed costs are present.
                if p.F > 0.0
                    val_inact = flow_profit(z, k, 0.0, p) + p.β * EV_itp(k_keep)
                    if val_inact > best_val
                        best_val = val_inact
                        best_i   = 0.0
                    end
                end

                V_new[ik, iz] = best_val
                i_star[ik, iz] = best_i
            end
        end

        diff = maximum(abs.(V_new .- V))
        copyto!(V, V_new)
        verbose && iter % 50 == 0 &&
            @printf("  VFI iter %4d | diff = %.2e\n", iter, diff)
    end

    if verbose
        diff <= tol ?
            @printf("  ✓ Converged in %d iters (diff=%.2e)\n", iter, diff) :
            @printf("  ✗ Did not converge after %d iters (diff=%.2e)\n", iter, diff)
    end

    return V, i_star
end

# ============================================================
# SECTION 4: STATIONARY DISTRIBUTION (Panel Simulation)
# ============================================================

# LRD empirical moments from Cooper & Haltiwanger (2006)
const DATA_MOMENTS = [0.122, 0.081, 0.104, 0.18, 0.014]
const MOMENT_NAMES = ["Avg Inv Rate",
                       "Inaction Rate (|i/k|<1%)",
                       "Frac Neg Investment",
                       "Pos Spike Rate (i/k>20%)",
                       "Neg Spike Rate (i/k<-20%)"]

# Simulate panel of firms using policy function and transition matrix to compute stationary distribution.
function simulate_panel(p::Params, g::Grids, i_star::Matrix{Float64};
                         N::Int=5000, T::Int=600, burn::Int=200, seed::Int=42)
    Random.seed!(seed)
    nk, nz = g.nk, g.nz

    i_itp  = interpolate((g.k_grid, g.z_grid), i_star, Gridded(Linear()))
    i_func(k, z) = i_itp(clamp(k, g.k_grid[1], g.k_grid[end]),
                          clamp(z, g.z_grid[1], g.z_grid[end]))

    k_vals = fill(g.k_grid[div(nk,2)], N)
    z_idx  = fill(div(nz,2), N)
    cum_Π  = cumsum(g.Π_z, dims=2)

    n_rec  = T - burn
    k_rec  = zeros(N, n_rec)
    z_rec  = zeros(N, n_rec)
    ir_rec = zeros(N, n_rec)

    for t in 1:T
        u = rand(N)
        @inbounds for n in 1:N
            row = z_idx[n]
            z_idx[n] = nz
            for iz in 1:nz
                if u[n] <= cum_Π[row, iz]; z_idx[n] = iz; break; end
            end
        end
        @inbounds for n in 1:N
            k   = k_vals[n]
            z   = g.z_grid[z_idx[n]]
            inv = i_func(k, z)
            k_next = clamp((1.0 - p.δ)*k + inv, g.k_grid[1], g.k_grid[end])
            if t > burn
                rec = t - burn
                k_rec[n,rec]  = k
                z_rec[n,rec]  = z
                ir_rec[n,rec] = inv / k
            end
            k_vals[n] = k_next
        end
    end

    return vec(k_rec), vec(z_rec), vec(ir_rec)
end

# Compute stationary distribution μ(k,z) by iterating on the distribution using the policy function and transition matrix until convergence.
function compute_moments(inv_rates::AbstractVector{Float64})
    return [mean(inv_rates),
            mean(abs.(inv_rates) .< 0.01),
            mean(inv_rates .< 0.0),
            mean(inv_rates .> 0.20),
            mean(inv_rates .< -0.20)]
end

function compute_aggregates(p::Params, g::Grids, i_star::Matrix{Float64};
                             N::Int=5000, T::Int=600, burn::Int=200)
    k_out, z_out, inv_rates = simulate_panel(p, g, i_star; N=N, T=T, burn=burn)

    K      = mean(k_out)
    h_vals = h_star.(z_out, k_out, Ref(p))
    H      = mean(h_vals)
    Y      = mean(z_out .* k_out .^ p.α .* h_vals .^ p.ν)

    pos_inv   = inv_rates .* k_out
    cost      = p.τ > 0 ? mean(p.τ .* max.(pos_inv, 0.0)) : 0.0
    cost_frac = cost / Y
    corr_kz   = cor(k_out, z_out)
    moments   = compute_moments(inv_rates)

    return (K=K, H=H, Y=Y, cost_frac=cost_frac, corr_kz=corr_kz,
            moments=moments, inv_rates=inv_rates, k_out=k_out, z_out=z_out)
end

# ============================================================
# SECTION 5: METHOD OF MOMENTS ESTIMATION
# ============================================================

# SMM objective function: distance between model and data moments for given θ.
function smm_objective(θ::Vector{Float64}; nk::Int=40, nz::Int=12)
    γ  = max(θ[1], 0.01)
    F  = max(θ[2], 0.0)
    ps = clamp(θ[3], 0.0, 1.0)
    try
        p = make_params(γ=γ, F=F, ps=ps)
        g = make_grids(p; nk=nk, nz=nz)
        _, i_star = solve_vfi(p, g; verbose=false, tol=1e-5, ni=200)
        _, _, ir  = simulate_panel(p, g, i_star; N=3000, T=400, burn=100)
        diff      = compute_moments(ir) .- DATA_MOMENTS
        return dot(diff, diff)
    catch e
        @warn "smm_objective error: $e"
        return 1e8
    end
end

# Nelder-Mead optimization to find θ that minimizes the SMM objective.
function smm_estimate(θ0::Vector{Float64}=[0.5, 0.02, 0.95];
                       nk::Int=40, nz::Int=12)
    println("Starting SMM estimation (Nelder-Mead)...")
    @printf("Initial θ = (γ=%.3f, F=%.4f, ps=%.3f)\n", θ0...)

    call_count = Ref(0)
    best_obj   = Ref(Inf)

    function obj_tracked(θ)
        val = smm_objective(θ; nk=nk, nz=nz)
        call_count[] += 1
        if val < best_obj[]
            best_obj[] = val
        end
        if call_count[] % 10 == 0
            γ, F, ps = max(θ[1],0.01), max(θ[2],0.0), clamp(θ[3],0.0,1.0)
            @printf("  eval %3d | best obj=%.6f | θ=(γ=%.3f, F=%.4f, ps=%.3f)\n",
                    call_count[], best_obj[], γ, F, ps)
        end
        return val
    end

    result = optimize(obj_tracked, θ0, NelderMead(),
                      Optim.Options(show_trace=false, iterations=300,
                                    x_tol=1e-4, f_tol=1e-6))

    θ_hat = Optim.minimizer(result)
    println("\n--- SMM Results ---")
    @printf("  γ  = %.4f\n  F  = %.4f\n  ps = %.4f\n", θ_hat...)
    @printf("  Objective: %.6f | Converged: %s | Evaluations: %d\n",
            Optim.minimum(result), Optim.converged(result) ? "yes" : "no",
            call_count[])

    return θ_hat, result
end

# ============================================================
# SECTION 6: FIGURES
# ============================================================

z_plot_idx(nz) = [max(1, round(Int,0.15*nz)), round(Int,0.5*nz),
                   min(nz, round(Int,0.85*nz))]

function plot_exploration(grids_list, i_list, labels)
    colors = [:royalblue, :darkorange, :forestgreen]
    pl = plot(layout=(1,3), size=(1400,420), link=:y,
              bottom_margin=5Plots.mm, left_margin=6Plots.mm)
    for (m, (g, i_mat, lab)) in enumerate(zip(grids_list, i_list, labels))
        for (ci, iz) in enumerate(z_plot_idx(g.nz))
            plot!(pl[m], g.k_grid, i_mat[:,iz], color=colors[ci], lw=2,
                  label="z=$(round(g.z_grid[iz],digits=2))")
        end
        hline!(pl[m], [0.0], color=:black, ls=:dot, lw=1, label="")
        title!(pl[m], lab); xlabel!(pl[m], "Capital k")
        m == 1 && ylabel!(pl[m], "Investment i*(k,z)")
    end
    return pl
end

function plot_policy(g::Grids, i_star::Matrix{Float64}, ttl::String)
    colors = [:royalblue, :darkorange, :forestgreen]
    pl = plot(title=ttl, xlabel="Capital k", ylabel="Investment i*(k,z)",
              size=(700,450), legend=:topleft)
    for (ci, iz) in enumerate(z_plot_idx(g.nz))
        plot!(pl, g.k_grid, i_star[:,iz], color=colors[ci], lw=2,
              label="z=$(round(g.z_grid[iz],digits=3))")
    end
    hline!(pl, [0.0], color=:black, ls=:dot, lw=1, label="i=0")
    return pl
end

function plot_value(g::Grids, V::Matrix{Float64})
    colors = [:royalblue, :darkorange, :forestgreen]
    pl = plot(title="Value Function V(k,z)", xlabel="Capital k", ylabel="V(k,z)",
              size=(700,450), legend=:bottomright)
    for (ci, iz) in enumerate(z_plot_idx(g.nz))
        plot!(pl, g.k_grid, V[:,iz], color=colors[ci], lw=2,
              label="z=$(round(g.z_grid[iz],digits=3))")
    end
    return pl
end

function plot_comparison(g_b::Grids, i_b::Matrix{Float64},
                          g_s::Grids, i_s::Matrix{Float64})
    iz = div(g_b.nz, 2)
    pl = plot(title="Policy: Baseline vs Subsidy (τ=0.10)",
              xlabel="Capital k", ylabel="Investment i*(k,z)",
              size=(750,480), legend=:topleft)
    plot!(pl, g_b.k_grid, i_b[:,iz], color=:royalblue, lw=2.5, label="Baseline (τ=0)")
    plot!(pl, g_s.k_grid, i_s[:,iz], color=:crimson, lw=2.5, ls=:dash,
          label="Subsidy (τ=0.10)")
    hline!(pl, [0.0], color=:black, ls=:dot, lw=1, label="")
    return pl
end

function plot_distributions(k_b, k_s, ir_b, ir_s)
    pl = plot(layout=(1,2), size=(1100,450),
              bottom_margin=5Plots.mm, left_margin=5Plots.mm)
    histogram!(pl[1], k_b,  bins=60, normalize=:pdf, alpha=0.55,
               color=:royalblue, label="Baseline")
    histogram!(pl[1], k_s,  bins=60, normalize=:pdf, alpha=0.55,
               color=:crimson, label="Subsidy τ=0.10")
    title!(pl[1], "Distribution of Capital"); xlabel!(pl[1], "k")
    histogram!(pl[2], ir_b, bins=80, normalize=:pdf, alpha=0.55,
               color=:royalblue, label="Baseline")
    histogram!(pl[2], ir_s, bins=80, normalize=:pdf, alpha=0.55,
               color=:crimson, label="Subsidy τ=0.10")
    title!(pl[2], "Distribution of Investment Rates")
    xlabel!(pl[2], "i/k"); xlims!(pl[2], -0.5, 0.8)
    return pl
end

function plot_moment_fit(m_model::Vector{Float64})
    labels = ["Avg i/k","Inaction","Neg Inv","Pos Spike","Neg Spike"]
    xs = 1:5
    pl = bar(xs .- 0.2, DATA_MOMENTS, bar_width=0.35, label="Data (LRD)",
             color=:royalblue, alpha=0.8, title="Moment Fit: Model vs Data",
             ylabel="Fraction", xticks=(xs, labels), size=(750,450),
             legend=:topright)
    bar!(pl, xs .+ 0.2, m_model, bar_width=0.35, label="Model",
         color=:darkorange, alpha=0.8)
    return pl
end

function save_figures(outdir, g0, i_conv, i_fix, i_irr,
                       g_cal, V_cal, i_cal, g_sub, i_sub,
                       agg_base, agg_sub, m_cal)
    isdir(outdir) || mkpath(outdir)
    figs = [
        ("fig1_exploration.png",
         plot_exploration([g0,g0,g0], [i_conv,i_fix,i_irr],
             ["Convex Only (F=0,ps=1)",
              "Fixed Costs (F=0.02,ps=1)",
              "Irreversibility (F=0.02,ps=0.95)"])),
        ("fig2a_policy_baseline.png",
         plot_policy(g_cal, i_cal, "Investment Policy (Calibrated Baseline)")),
        ("fig2b_policy_subsidy.png",
         plot_policy(g_sub, i_sub, "Investment Policy (Subsidy τ=0.10)")),
        ("fig3_value_function.png",    plot_value(g_cal, V_cal)),
        ("fig4_policy_comparison.png", plot_comparison(g_cal, i_cal, g_sub, i_sub)),
        ("fig5_distributions.png",
         plot_distributions(agg_base.k_out, agg_sub.k_out,
                            agg_base.inv_rates, agg_sub.inv_rates)),
        ("fig6_moment_fit.png",        plot_moment_fit(m_cal)),
    ]
    for (fname, pl) in figs
        savefig(pl, joinpath(outdir, fname))
        println("  ✓ $fname")
    end
    println("All figures saved to: $outdir")
end

# ============================================================
# SECTION 7: MAIN
# ============================================================

function main()
    println("="^65)
    println("  Heterogeneous Firm Model — Quantitative Economics Final Project")
    println("="^65)

    # ----------------------------------------------------------
    # PART A: Model Exploration
    # Solve three variants to understand the role of each friction.
    # ----------------------------------------------------------
    println("\n" * "─"^50)
    println("PART A: Model Exploration")
    println("─"^50)

    p_conv = make_params(γ=0.5, F=0.0,  ps=1.0)
    p_fix  = make_params(γ=0.5, F=0.02, ps=1.0)
    p_irr  = make_params(γ=0.5, F=0.02, ps=0.95)
    g0     = make_grids(p_conv; nk=50, nz=15)

    println("\n[A.1] Convex costs only (γ=0.5, F=0, ps=1)")
    V_conv, i_conv = solve_vfi(p_conv, g0)

    println("\n[A.2] Adding fixed costs (γ=0.5, F=0.02, ps=1)")
    V_fix, i_fix = solve_vfi(p_fix, g0)

    println("\n[A.3] Adding irreversibility (γ=0.5, F=0.02, ps=0.95)")
    V_irr, i_irr = solve_vfi(p_irr, g0)

    println("\nInvestment moments by variant:")
    @printf("%-30s %8s %8s %8s %8s %8s\n",
            "Model","AvgI/K","Inact","NegInv","+Spike","-Spike")
    println("─"^72)
    for (lab, p_v, i_v) in [("Convex (F=0,ps=1)",    p_conv, i_conv),
                              ("Fixed  (F=0.02,ps=1)", p_fix,  i_fix),
                              ("Irrev. (F=0.02,ps=.95)",p_irr, i_irr)]
        _, _, ir = simulate_panel(p_v, g0, i_v; N=3000, T=400, burn=100)
        m = compute_moments(ir)
        @printf("%-30s %8.3f %8.3f %8.3f %8.3f %8.3f\n", lab, m...)
    end
    @printf("%-30s %8.3f %8.3f %8.3f %8.3f %8.3f\n", "Data (LRD)", DATA_MOMENTS...)

    # ----------------------------------------------------------
    # PART B: SMM Calibration
    # ----------------------------------------------------------
    println("\n" * "─"^50)
    println("PART B: Method of Moments Calibration")
    println("─"^50)

    θ_hat, _ = smm_estimate([0.5, 0.02, 0.95]; nk=40, nz=12)
    γ̂, F̂, p̂s = θ_hat

    println("\nSolving calibrated model on fine grid (nk=50, nz=15)...")
    p_cal = make_params(γ=γ̂, F=F̂, ps=p̂s, τ=0.0)
    g_cal = make_grids(p_cal; nk=50, nz=15)
    V_cal, i_cal = solve_vfi(p_cal, g_cal)

    _, _, ir_cal = simulate_panel(p_cal, g_cal, i_cal; N=5000, T=600, burn=200)
    m_cal = compute_moments(ir_cal)

    println("\nMoment fit:")
    @printf("%-28s %8s %8s %8s\n", "Moment", "Data", "Model", "|Error|")
    println("─"^55)
    for (n, d, m) in zip(MOMENT_NAMES, DATA_MOMENTS, m_cal)
        @printf("%-28s %8.3f %8.3f %8.3f\n", n, d, m, abs(m-d))
    end

    # ----------------------------------------------------------
    # PART C: Policy Analysis — Investment Subsidy τ = 0.10
    # ----------------------------------------------------------
    println("\n" * "─"^50)
    println("PART C: Policy Analysis (τ = 0.10)")
    println("─"^50)

    println("\nSolving subsidized model...")
    p_sub = make_params(γ=γ̂, F=F̂, ps=p̂s, τ=0.10)
    g_sub = make_grids(p_sub; nk=50, nz=15)
    V_sub, i_sub = solve_vfi(p_sub, g_sub)

    println("\nComputing stationary distributions...")
    agg_base = compute_aggregates(p_cal, g_cal, i_cal)
    agg_sub  = compute_aggregates(p_sub, g_sub, i_sub)

    println("\nAggregates (% change):")
    @printf("%-10s %12s %12s %10s\n", "Variable","Baseline","Subsidy","% Chg")
    println("─"^48)
    for (var, vb, vs) in [("K",agg_base.K,agg_sub.K),
                           ("H",agg_base.H,agg_sub.H),
                           ("Y",agg_base.Y,agg_sub.Y)]
        @printf("%-10s %12.4f %12.4f %9.2f%%\n", var, vb, vs, 100*(vs/vb-1))
    end

    println("\nCross-sectional patterns:")
    @printf("  Corr(k,z) baseline: %.4f\n", agg_base.corr_kz)
    @printf("  Corr(k,z) subsidy:  %.4f\n", agg_sub.corr_kz)

    println("\nSubsidy cost:")
    @printf("  Cost/Y = %.4f (%.2f%%)  |  ΔY/Y = %.2f%%\n",
            agg_sub.cost_frac, 100*agg_sub.cost_frac,
            100*(agg_sub.Y/agg_base.Y - 1))

    println("\nInvestment moments:")
    @printf("%-28s %8s %8s %8s\n","Moment","Data","Base","Subsidy")
    println("─"^56)
    for (n, d, mb, ms) in zip(MOMENT_NAMES, DATA_MOMENTS,
                                agg_base.moments, agg_sub.moments)
        @printf("%-28s %8.3f %8.3f %8.3f\n", n, d, mb, ms)
    end

    # ----------------------------------------------------------
    # PART D: Grid Sensitivity
    # ----------------------------------------------------------
    println("\n" * "─"^50)
    println("PART D: Grid Sensitivity Check")
    println("─"^50)

    println("\nCoarse grid (nk=30, nz=10):")
    g_c  = make_grids(p_cal; nk=30, nz=10)
    _, i_c = solve_vfi(p_cal, g_c; tol=1e-5, ni=150)
    _, _, ir_c = simulate_panel(p_cal, g_c, i_c; N=3000, T=400, burn=100)

    println("\nMoment sensitivity:")
    @printf("%-28s %8s %8s\n","Moment","Coarse","Fine")
    println("─"^46)
    for (n, mc, mf) in zip(MOMENT_NAMES, compute_moments(ir_c), m_cal)
        @printf("%-28s %8.3f %8.3f\n", n, mc, mf)
    end

    # ----------------------------------------------------------
    # PART E: Figures
    # ----------------------------------------------------------
    println("\n" * "─"^50)
    println("PART E: Saving Figures")
    println("─"^50)

    outdir = joinpath(pwd(), "figures")
    save_figures(outdir, g0, i_conv, i_fix, i_irr,
                  g_cal, V_cal, i_cal, g_sub, i_sub,
                  agg_base, agg_sub, m_cal)

    println("\n" * "="^65)
    println("Done. Figures saved to: $outdir")
    println("="^65)
end

main()
