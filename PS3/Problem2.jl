using Distributions
using Random
using Statistics
using Plots

# O ∈ {-1, 0, 1}  (dead, not yet bloomed, in bloom)
const O_DEAD  = 1   # O = -1
const O_NOT   = 2   # O =  0
const O_BLOOM = 3   # O =  1

Base.@kwdef struct Params
    T::Int
    Smax::Int
    λ::Float64
    δ::Float64
    ρ::Float64
    c0::Float64
    c1::Float64
    c2::Float64
    α::Float64
    ψ::Float64
    κ::Float64
    θ::Float64
    ω::Float64
end

# Rewards
anxiety_cost(t, par::Params) = par.c0 + par.c1 * t + par.c2 * t^2

function flow_reward(Oidx::Int, t::Int, par::Params)
    if Oidx == O_NOT
        return -anxiety_cost(t, par)
    elseif Oidx == O_BLOOM
        return par.α
    else # O_DEAD
        return -anxiety_cost(t, par) - par.ψ
    end
end

function terminal_payoff(Oidx::Int, p::Float64, par::Params)
    if Oidx == O_NOT
        return -par.κ * p
    elseif Oidx == O_BLOOM
        return  par.θ * p
    else # O_DEAD
        return -par.ω * p
    end
end

# Bellman solve
"""
solve_bellman(par, p) -> V, policy

V[t, Oidx, S+1] is value at day t, status Oidx, stress S (0..Smax).
policy[t, S+1] is optimal action when O=0:
  0 = Wait (W)
  1 = Apply fertilizer (A)
"""
function solve_bellman(par::Params, p::Float64)
    T, Smax = par.T, par.Smax
    V = zeros(Float64, T + 1, 3, Smax + 1)        
    policy = fill(0, T, Smax + 1)                 

    pois = Poisson(par.ρ)

    for Oidx in 1:3, S in 0:Smax
        V[T, Oidx, S+1] = flow_reward(Oidx, T, par) + terminal_payoff(Oidx, p, par)
    end

    for t in (T-1):-1:1
        for S in 0:Smax
            # Absorbing states: bloom/dead
            for Oidx in (O_BLOOM, O_DEAD)
                V[t, Oidx, S+1] = flow_reward(Oidx, t, par) + V[t+1, Oidx, S+1]
            end

            # Decision state: O=0
            r0 = flow_reward(O_NOT, t, par)

            # Action W (wait): bloom w.p. λ, stress unchanged
            EV_W = par.λ * V[t+1, O_BLOOM, S+1] + (1 - par.λ) * V[t+1, O_NOT, S+1]

            # Action A (apply):
            EV_A = 0.0
            if S == Smax
                EV_A = V[t+1, O_DEAD, Smax+1]
            else
                maxk = Smax - S
                probs = pdf.(pois, 0:maxk)                    
                tail  = max(0.0, 1.0 - sum(probs))            

                tmp = 0.0
                for k in 0:maxk
                    pk = probs[k+1]
                    S2 = S + k
                    tmp += pk * (
                        (par.λ + par.δ) * V[t+1, O_BLOOM, S2+1] +
                        (1 - par.λ - par.δ) * V[t+1, O_NOT,   S2+1]
                    )
                end
                tmp += tail * V[t+1, O_DEAD, Smax+1]
                EV_A = tmp
            end

            # Choose best
            if EV_A > EV_W
                V[t, O_NOT, S+1] = r0 + EV_A
                policy[t, S+1] = 1
            else
                V[t, O_NOT, S+1] = r0 + EV_W
                policy[t, S+1] = 0
            end
        end
    end

    return V, policy
end

# Simulation under a policy
"""
simulate_paths(par, policy; N, seed)

Simulates N paths from (t=1, O=0, S=0) following policy[t,S].

Returns:
  final_states          :: Vector{Int} in {O_DEAD,O_NOT,O_BLOOM}
  fertil_counts         :: Vector{Int} number of fertilizer applications per path
  bloom_prob_by_day     :: Vector{Float64} length T, estimates P(O_t=1)
"""
function simulate_paths(par::Params, policy; N::Int=1000, seed::Int=1234)
    rng = MersenneTwister(seed)
    pois = Poisson(par.ρ)

    final_states = Vector{Int}(undef, N)
    fertil_counts = zeros(Int, N)
    bloom_count_by_day = zeros(Int, par.T)

    for n in 1:N
        Oidx = O_NOT
        S = 0

        for t in 1:par.T
            #record whether bloomed at day t
            if Oidx == O_BLOOM
                bloom_count_by_day[t] += 1
            end

            #no action at t = T
            if t == par.T
                break
            end

            #if absorbing, nothing changes
            if Oidx != O_NOT
                continue
            end

            a = policy[t, S+1]  # 0=Wait 1=Apply

            if a == 0
                if rand(rng) < par.λ
                    Oidx = O_BLOOM
                end
            else
                # fertilizer
                fertil_counts[n] += 1

                if S == par.Smax
                    Oidx = O_DEAD
                    S = par.Smax
                else
                    k = rand(rng, pois)
                    if S + k > par.Smax
                        Oidx = O_DEAD
                        S = par.Smax
                    else
                        S += k
                        if rand(rng) < (par.λ + par.δ)
                            Oidx = O_BLOOM
                        else
                            Oidx = O_NOT
                        end
                    end
                end
            end
        end

        final_states[n] = Oidx
    end

    bloom_prob_by_day = bloom_count_by_day ./ N
    return final_states, fertil_counts, bloom_prob_by_day
end

# Plot helpers (prevents blank/duplicate plots)
gr()
Plots.closeall()

function show_or_save!(plt, filename; do_display::Bool=true, do_save::Bool=false)
    if do_save
        savefig(plt, filename)
    end
    if do_display
        display(plt)
    end
    return nothing
end

#given parameters in the prompt
par = Params(
    T=20, Smax=10,
    λ=0.05, δ=0.25, ρ=5.0,
    c0=0.5, c1=0.5, c2=0.1,
    α=5.0, ψ=10.0,
    κ=0.5, θ=2.0, ω=2.5
)

p0 = 100.0

V, policy = solve_bellman(par, p0)

# 4(a) 
a1 = policy[1, 0+1]
println("4(a) Optimal action at (t=1, O=0, S=0): ", a1 == 0 ? "Wait (W)" : "Apply fertilizer (A)")

# 4(b) 
stress = 0:par.Smax
tlist = [1, 5, 10, 15, 19]

plt_pol = plot(
    title  = "Optimal policy (O=0) vs stress",
    xlabel = "Stress S",
    ylabel = "Action (0=Wait, 1=Apply)",
    legend = :topright
)
for t in tlist
    plot!(plt_pol, stress, policy[t, :], marker=:circle, label="t=$t");
end
show_or_save!(plt_pol, "policy_by_stress.png"; do_display=true, do_save=false)

# 4(c)
Vstart = V[1, O_NOT, 0+1]
println("4(c) Expected total utility starting from (1, O=0, S=0): ", Vstart)

# 4(d,e,f)
Nsim = 1000
final_states, fertil_counts, bloom_prob_by_day = simulate_paths(par, policy; N=Nsim, seed=2026)

frac_bloom = mean(final_states .== O_BLOOM)
frac_never = mean(final_states .== O_NOT)
frac_dead  = mean(final_states .== O_DEAD)

println("4(d) Fractions from simulation (N=$Nsim):")
println("     (i) successful bloom:  ", frac_bloom)
println("     (ii) never blooming:   ", frac_never)
println("     (iii) dying:           ", frac_dead)

avg_fertil = mean(fertil_counts)
println("4(e) Average fertilizer applications per path: ", avg_fertil, " (T = ", par.T, ")")

# 4(f)
days = 1:par.T
plt_bloom = plot(
    days, bloom_prob_by_day,
    marker=:circle,
    title  = "Cumulative bloom probability under optimal policy",
    xlabel = "Day t",
    ylabel = "P(O_t = 1)",
    label  = "P(bloom by t)"
)
show_or_save!(plt_bloom, "bloom_probability.png"; do_display=true, do_save=false)

# 4(g)
prices = 50.0:1.0:200.0
Vstarts = Vector{Float64}(undef, length(prices))

for (i, p) in enumerate(prices)
    Vp, _ = solve_bellman(par, p)
    Vstarts[i] = Vp[1, O_NOT, 1]  
end

plt_p = plot(
    prices, Vstarts,
    title  = "Expected utility V1(0,0) vs price p",
    xlabel = "Price p",
    ylabel = "V1(0,0)",
    label  = "V1(0,0)"
)
hline!(plt_p, [0.0], linestyle=:dash, label="0");
show_or_save!(plt_p, "value_vs_price.png"; do_display=true, do_save=false)

# Find approximate crossing where value becomes negative
cross_p = NaN
for i in 2:length(prices)
    v1, v2 = Vstarts[i-1], Vstarts[i]
    if (v1 ≤ 0 && v2 > 0) || (v1 ≥ 0 && v2 < 0)
        p_lo, p_hi = prices[i-1], prices[i]
        cross_p = p_lo + (0 - v1) * (p_hi - p_lo) / (v2 - v1)
        break
    end
end

if isnan(cross_p)
    println("4(g) No sign change on [50,200]. Min=", minimum(Vstarts), " Max=", maximum(Vstarts))
else
    if Vstarts[1] < 0
        println("4(g) Utility becomes negative for p < ", cross_p, " (crossing at p≈", cross_p, ")")
    else
        println("4(g) Utility becomes negative for p > ", cross_p, " (crossing at p≈", cross_p, ")")
    end
end
