# Problem 2 – Simulated Method of Moments (SMM)


using Random, Statistics, Optim, Plots

# 1st define function to compute excess kurtosis

function excess_kurtosis(x::AbstractVector)
    μ = mean(x)
    σ = std(x)
    z = (x .- μ) ./ σ
    return mean(z.^4) - 3.0
end

# True parameter values and model setup

ρ_true  = 0.90
p_true  = 0.80
σL      = 0.10   # low volatility
σH      = 0.30   # high volatility
T_total = 500    # total simulated periods (before burn-in cut)
burn_in = 100    # discard first 100 obs

# 1. Generate "observed" data from the true model
#
# log y_{t+1} = ρ log y_t + ε_{t+1}
# with ε drawn from:
#   N(0, σL^2) with prob p
#   N(0, σH^2) with prob 1-p

function simulate_observed_data(T_total::Int;
                                ρ=ρ_true,
                                p=p_true,
                                σL=σL,
                                σH=σH,
                                seed=2024,
                                burn_in=burn_in)

    # Set seed for reproducibility
    Random.seed!(seed)

    # We store log y_t in a vector, starting from log y_1 = 0
    logy = zeros(Float64, T_total)
    # logy[1] = 0 already

    # Simulate forward
    for t in 1:(T_total - 1)
        # Decide whether we are in low-volatility or high-volatility regime
        is_low = rand() < p
        σ = is_low ? σL : σH

        # Draw innovation ε ~ N(0, σ^2)
        ε = σ * randn()

        # AR(1) update for log y
        logy[t + 1] = ρ * logy[t] + ε
    end

    # Discard the first "burn_in" observations
    return logy[(burn_in + 1):end]
end

# Generate observed data (length will be 500 - 100 = 400)
observed_logy = simulate_observed_data(T_total)
println("Length of observed data = ", length(observed_logy))

# 2. Compute moments from data:
#    m_b1 = std(log y_t)
#    m_b2 = corr(log y_t, log y_{t-1})
#    m_b3 = excess kurtosis of Δ log y_t

function compute_moments(logy::AbstractVector)
    # Standard deviation of log y_t
    m1 = std(logy)

    # Lag-1 autocorrelation: corr(log y_t, log y_{t-1})
    m2 = cor(logy[2:end], logy[1:end-1])

    # Changes Δ log y_t
    Δlogy = diff(logy)

    # Excess kurtosis of Δ log y_t
    m3 = excess_kurtosis(Δlogy)

    return (m1, m2, m3)
end

mb1, mb2, mb3 = compute_moments(observed_logy)
println("Observed moments:")
println("  std(log y_t)             = ", mb1)
println("  corr(log y_t, log y_{t-1}) = ", mb2)
println("  excess kurtosis(Δlog y_t) = ", mb3)

# 3. simulate_model(θ, T, σL, σH)
#    - θ = (ρ, p)
#    - simulate T + burn_in periods
#    - return only last T obs (burn-in dropped)
function simulate_model(θ::AbstractVector,
                        T::Int,
                        σL::Float64,
                        σH::Float64;
                        burn_in::Int = 100)

    ρ, p = θ
    T_total = T + burn_in

    logy = zeros(Float64, T_total)  # log y_1 = 0

    for t in 1:(T_total - 1)
        is_low = rand() < p
        σ = is_low ? σL : σH
        ε = σ * randn()
        logy[t + 1] = ρ * logy[t] + ε
    end

    # Return last T observations after burn-in
    return logy[(burn_in + 1):end]
end

# 4. SMM objective function Q(θ)

function smm_objective(θ::Vector{Float64},
                       observed_data::Vector{Float64},
                       σL::Float64,
                       σH::Float64,
                       S::Int)

    ρ, p = θ

    # Hard penalty if θ outside allowed region
    if ρ <= 0.5 || ρ >= 0.99 || p <= 0.5 || p >= 0.95
        return 1e6
    end

    # Seed depends on parameters => deterministic Q(θ)
    seed = round(Int, 10_000 * (ρ + 2.0 * p))
    Random.seed!(seed)

    # Observed moments
    mb1, mb2, mb3 = compute_moments(observed_data)
    T = length(observed_data)

    # Initialize sums of simulated moments
    sim_m1 = 0.0
    sim_m2 = 0.0
    sim_m3 = 0.0

    # Loop over simulations
    for s in 1:S
        sim_logy = simulate_model(θ, T, σL, σH)
        m1, m2, m3 = compute_moments(sim_logy)
        sim_m1 += m1
        sim_m2 += m2
        sim_m3 += m3
    end

    # Averages over simulations
    mbar1 = sim_m1 / S
    mbar2 = sim_m2 / S
    mbar3 = sim_m3 / S

    # SMM objective: sum of squared differences
    Q = (mb1 - mbar1)^2 + (mb2 - mbar2)^2 + (mb3 - mbar3)^2
    return Q
end

# 5. Minimize Q(θ) using Optim.jl

S = 100                          # number of simulations in SMM
θ0 = [0.85, 0.70]                # initial guess (ρ, p)
lower = [0.5, 0.5]               # lower bounds
upper = [0.99, 0.95]             # upper bounds

# Wrapper so Optim sees θ as an abstract vector
obj(θ) = smm_objective(collect(θ), observed_logy, σL, σH, S)

# Use box-constrained Nelder-Mead
res = optimize(obj, lower, upper, θ0, Fminbox(NelderMead()))

θ_hat = Optim.minimizer(res)
ρ_hat, p_hat = θ_hat

println("\n SMM estimation results")
println("Estimated ρ = ", ρ_hat, "   (true = ", ρ_true, ")")
println("Estimated p = ", p_hat, "   (true = ", p_true, ")")
println("Minimum Q(θ̂) = ", Optim.minimum(res))

# 6. Compare observed vs simulated time series and histograms

# Simulate one long series from estimated parameters
T_compare = 500
Random.seed!(1234)  # seed for comparison simulation
sim_logy_hat = simulate_model(θ_hat, T_compare, σL, σH)

# Time series: first 200 periods (make sure both vectors are long enough)
T_plot = 200
ts_obs = observed_logy[1:T_plot]
ts_sim = sim_logy_hat[1:T_plot]

plot(1:T_plot, ts_obs, label = "Observed",
     xlabel = "t", ylabel = "log y_t",
     title = "Observed vs simulated log income (first 200 periods)")
plot!(1:T_plot, ts_sim, label = "Simulated (θ̂)")

# Histogram of Δlog y_t: observed vs simulated
Δobs = diff(observed_logy)
Δsim = diff(sim_logy_hat)

plot()  # new plot
histogram(Δobs, normalize = :pdf, label = "Observed",
          xlabel = "Δ log y_t", ylabel = "Density",
          title = "Histogram of Δ log y_t: observed vs simulated")
histogram!(Δsim, normalize = :pdf, label = "Simulated (θ̂)", alpha = 0.5)

println("\nDone. Check the two plots for visual comparison.")