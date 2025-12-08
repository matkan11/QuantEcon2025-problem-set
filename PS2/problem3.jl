#Problem 3: Transition dynamics in a neoclassical growth model

using NLsolve, Plots, LinearAlgebra

#Define parameters
β = 0.96
α = 0.33
A = 1.0
δ = 0.1
T = 100 

#solving for gamma = 0.5  first
γ = 0.5

#Calculating steady state with formulas from task
#solving euler formula for k bar
k_bar = (( ( 1/β) - (1-δ)) / (α * A))^(1/(α-1))

#solvign c bar 
c_bar = A * k_bar^α - δ * k_bar

#Initial capital stock
k0 = 0.5 * k_bar

println("Target capital: ", k_bar)
println("Target consumption: ", c_bar)


#Equation system 

params = (β, α, A, δ, γ, T, k0, k_bar, c_bar)

function transition_equations(x, params)
    β, α, A, δ, γ, T, k0, k_bar, c_bar = params
    
    #make vector residuals of length 2T+1
    residuals = zeros(2T + 1)

    #unpack x into k and c
    C = x[1:T+1]
    K = x[T+2:end]

    #loop over time periods

    for t in 0:(T-1)
        idx = t + 1

        c_now = C[idx]
        c_next = C[idx + 1]
        k_next = K[idx]

        if t == 0 
            k_now = k0
        else 
            k_now = K[t]
        end

        #Euler equation 
        R = α * A * k_next^(α - 1) + (1 - δ)
        residuals[idx] = c_now^(-γ) - β * R * c_next^(-γ)

        #Capital accumulation equation
        Output = A * k_now^α
        residuals[T + idx] = k_next - ((1-δ)*k_now + Output - c_now)
    end

    #Terminal conditions
    residuals[end] = C[end] - c_bar

    return residuals
end

#Solving the system 

println("Solving for γ = ", γ)

#Initial guess
guess_c = fill(c_bar, T + 1)
guess_k = range(k0, k_bar, length=T + 1)[2:end]
x0 = vcat(guess_c, guess_k)

#Solve using NLsolve

function f!(F, x)
    F .= transition_equations(x, params)
end

solution = nlsolve(f!, x0)

println("Converged? ", converged(solution))
println("Residual norm: ", norm(solution.zero))

#Extract solution for later plotting 
results = solution.zero
C_sol_05 = results[1:T+1]
K_sol_05 = vcat(k0, results[T+2:end])
        
#Solvign for gamma = 2.0
γ_20 = 2.0
params_20 = (β, α, A, δ, γ_20, T, k0, k_bar, c_bar)

#New function for γ = 2.0
function f2!(F, x)
    F .= transition_equations(x, params_20)
end

solution_20 = nlsolve(f2!, x0)

println("Converged? ", converged(solution_20))
println("Residual norm: ", norm(solution_20.zero))

#Extract solution for later plotting
results_20 = solution_20.zero
C_sol_20 = results_20[1:T+1]
K_sol_20 = vcat(k0, results_20[T+2:end])



#Calculating Output 
Y_05 = A .* K_sol_05 .^ α
Y_20 = A .* K_sol_20 .^ α

#Calculating Investment
I_05 = Y_05 .- C_sol_05
I_20 = Y_20 .- C_sol_20

#Calculating steady state 
Y_ss = A * k_bar^α
I_ss = Y_ss - c_bar

#steady state consumption and investment rates
cy_target = c_bar / Y_ss
iy_target = I_ss / Y_ss

#Plots

time_axis = 0:T

#Plot Capital

p1 = plot(time_axis, K_sol_05, label="γ=0.5", color=:blue, lw=2, title="Capital Stock")
plot!(p1, time_axis, K_sol_20, label="γ = 2.0", color=:red, linestyle=:dash, lw=2)
hline!(p1, [k_bar], label="Steady State", color=:black, linestyle=:dot)
ylabel!("Capital")

#Plot Consumption
p2 = plot(time_axis, C_sol_05 ./ Y_05, label="γ = 0.5", color=:blue, lw=2, title="Consumption Rate (c/y)")
plot!(p2, time_axis, C_sol_20 ./ Y_20, label="γ = 2.0", color=:red, linestyle=:dash, lw=2)
hline!(p2, [cy_target], label="Steady State", color=:black, linestyle=:dot)
ylabel!("Share of Output")

#Plot Investment
p3 = plot(time_axis, I_05 ./ Y_05, label="γ = 0.5", color=:blue, lw=2, title="Investment Rate (i/y)")
plot!(p3, time_axis, I_20 ./ Y_20, label="γ = 2.0", color=:red, linestyle=:dash, lw=2)
hline!(p3, [iy_target], label="Steady State", color=:black, linestyle=:dot)
ylabel!("Share of Output")
xlabel!("Time")

#Display plots
plot(p1, p2, p3, layout=(3,1), size=(700,900))

