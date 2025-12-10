# problem 1, Hubert Mgłowski and Mikołaj Dessaeur

using QuadGK
using Roots
using Plots

#lognormal density function
function lognormal_pdf(r, μ, σ)
    r <= 0 && return 0.0
    return 1 / (r * σ * sqrt(2π)) * exp(-(log(r) - μ)^2 / (2σ^2))
end

#marginal utility
function u_prime(wealth, γ)
    γ == 0.0 && return 1.0             
    return wealth^(-γ)                 
end

#foc integral
function foc_integral(ω, W, Rf, γ, μ, σ)
    integrand(r) = (r - Rf) *
                   u_prime(W * (ω * r + (1 - ω) * Rf), γ) *
                   lognormal_pdf(r, μ, σ)

    val, _ = quadgk(integrand, 0.0, Inf; rtol = 1e-8, atol = 1e-10)
    return val
end

#check for y=0
function check_gamma_zero(W, Rf, μ, σ)
    γ = 0.0
    num, _ = quadgk(r -> (r - Rf) * lognormal_pdf(r, μ, σ),
                    0.0, Inf; rtol = 1e-10, atol = 1e-12)
    analytic = exp(μ + σ^2 / 2) - Rf

    println("Check for γ = 0:")
    println(" numerical integral = $num")
    println(" analytic value     = $analytic")
    println(" difference         = $(num - analytic)")
end

# Optimal portfolio share ω*
function optimal_portfolio(W, Rf, γ, μ, σ;
                           ω_min::Float64 = 0.0,
                           ω_max::Float64 = 1.0)

    f(ω) = foc_integral(ω, W, Rf, γ, μ, σ)

    f_min = f(ω_min)
    f_max = f(ω_max)

    if f_min * f_max < 0
        # interior root
        ω_star = find_zero(f, (ω_min, ω_max); xtol = 1e-6)
        return ω_star
    else
        # no sign change: derivative does not vanish on [ω_min, ω_max]
        # concavity ⇒ optimum at one of the boundaries
        if f_min > 0 && f_max > 0
            return ω_max        # EU increasing in ω
        elseif f_min < 0 && f_max < 0
            return ω_min        # EU decreasing in ω
        else
            # extremely flat / numerical noise
            return abs(f_min) < abs(f_max) ? ω_min : ω_max
        end
    end
end

#compute ω*(γ) over a grid of γ's
function omega_vs_gamma(W, Rf, μ, σ;
                        γ_min = 0.1, γ_max = 10.0, n = 50,
                        ω_min = 0.0, ω_max = 1.0)

    γ_vals = collect(range(γ_min, γ_max; length = n))
    ω_vals = similar(γ_vals)

    for (i, γ) in enumerate(γ_vals)
        ω_vals[i] = optimal_portfolio(W, Rf, γ, μ, σ;
                                      ω_min = ω_min, ω_max = ω_max)
        println("γ = $(round(γ, digits = 3)), ω* = $(ω_vals[i])")
    end

    return γ_vals, ω_vals
end

#apply the parameters

W_ex  = 1.0
Rf_ex = 1.02
γ_ex  = 3.0
μ_ex  = 0.05
σ_ex  = 0.1

#1.1 check gamma=0
check_gamma_zero(W_ex, Rf_ex, μ_ex, σ_ex)

#1.2 optimal share ω* for γ = $γ_ex is $ω_star"
ω_star = optimal_portfolio(W_ex, Rf_ex, γ_ex, μ_ex, σ_ex;
                           ω_min = 0.0, ω_max = 1.0)
println("Optimal share ω* for γ = $γ_ex is $ω_star")

#how ω* variates
γ_grid, ω_grid = omega_vs_gamma(W_ex, Rf_ex, μ_ex, σ_ex;
                                γ_min = 0.1, γ_max = 10.0, n = 80,
                                ω_min = 0.0, ω_max = 1.0)

plt = plot(γ_grid, ω_grid,
           xlabel = "Risk aversion γ",
           ylabel = "Optimal risky share ω*",
           title  = "Optimal portfolio share ω* as a function of γ",
           legend = false)