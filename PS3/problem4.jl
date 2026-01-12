using Parameters, Plots, LinearAlgebra, Statistics, QuantEcon, Random, Distributions

#defining human capital model

@with_kw struct HumanCapitalModel
    α::Float64 = 0.1
    f_bar::Float64 = 1.2
    δ::Float64 = 0.1
    γ::Float64 = 1.5
    β::Float64 = 0.95
    ρ::Float64 = 0.98
    σ_eps::Float64 = 0.15
    ψ::Float64 = 0.5
    
    #education
    e_grid::Vector{Float64} = collect(0.0:0.01:1.0)

    #human capital (h_max 14, allowing safety margin)
    h_step::Float64 = 0.01
    h_max::Float64 = 14
    h_grid::Vector{Float64} = collect(0.0:h_step:h_max)
    n_h::Int = length(h_grid)

    #wage shock
    n_w::Int = 7
    mc::MarkovChain = rouwenhorst(n_w, ρ, σ_eps)
    #because process is in log
    w_grid::Vector{Float64} = exp.(mc.state_values)
    P::Matrix{Float64} = mc.p

    #functions:
    func_f::Function = h -> min(h^α + 0.1, f_bar)
    #utility function
    u::Function = c -> (c^(1-γ))/(1-γ)
end

model = HumanCapitalModel()

#Bellman operator T

function T_operator(V_old, model)
    @unpack e_grid, h_grid, w_grid, P, func_f, u, ψ, β, δ, h_step, n_h, n_w = model

    V_new = similar(V_old)
    #matrix for storage
    policy_e = zeros(n_h, n_w)

   #loop for state_values
    for (i_w, w) in enumerate(w_grid)
        for (i_h, h) in enumerate(h_grid)

            max_val = -Inf
            best_e = 0.0
            
            #e choices
            for e in e_grid

                #current I&C 
                y = w * func_f(h) * (1-e)
                c = max(1e-10, y)

                #current utility
                current_util = u(c) - ψ * e 

                #next period HC 
                h_next_val = h + e - δ

                #finding next h index
                i_h_next = round(Int, h_next_val / h_step) + 1
                i_h_next = clamp(i_h_next, 1, n_h)

                #expected continuation value
                expected_v = 0.0
                for i_w_next in 1:n_w
                    expected_v += P[i_w, i_w_next] * V_old[i_h_next, i_w_next]
                end
                
                #total value:
                total_value = current_util + β * expected_v

                #condintion for validity
                if  total_value > max_val
                    max_val = total_value
                    best_e = e
                end
            end

            V_new[i_h, i_w] = max_val
            policy_e[i_h, i_w] = best_e
        end
    end
    
    return V_new, policy_e
end


#VALUE FUNCTION ITERATION

function solve_model(model; tol=1e-5, maxiter=2000)
    @unpack n_h, n_w = model

    V = zeros(n_h, n_w)
    policy = zeros(n_h, n_w)

    iter = 0
    diff = tol + 1.0

    #starting VFI
    while diff > tol && iter < maxiter
        iter += 1
        V_new, policy_new = T_operator(V, model)
        diff = maximum(abs.(V_new - V))
        V = V_new 
        policy = policy_new

        if iter % 100 == 0
            println("Iter: $iter, Diff: $diff")
        end
    end
    println("Converged in $iter iterations.")
    return V, policy
end

#running solver
V_star, e_star = solve_model(model)



#               PLOTS  



#wage setup - low, medium, high

w_indices = [1, 4, 7]
labels = ["Low w", "Med w", "High w"]

# a  Optimal education choice 
p1 = plot(title="Optimal Education e*(h,w)", xlabel="Human Capital (h)", ylabel="Education Time (e)")
for (j, i_w) in enumerate(w_indices)
    plot!(p1, model.h_grid, e_star[:, i_w], label=labels[j], linewidth=2)
end


# b    value function
p2 = plot(title="Value Function V(h,w)", xlabel="h", ylabel="Value")
for (j, i_w) in enumerate(w_indices)
    plot!(p2, model.h_grid, V_star[:, i_w], label=labels[j], linewidth=2)
end


#c  Consumption (solving consumption from polciy f)
c_star = zeros(model.n_h, model.n_w)
for i_w in 1:model.n_w
    for i_h in 1:model.n_h
        w = model.w_grid[i_w]
        h = model.h_grid[i_h]
        e = e_star[i_h, i_w]
        c_star[i_h, i_w] = w * model.func_f(h) * (1 - e)
    end
end
            
p3 = plot(title="Consumption c*(h,w)", xlabel="h", ylabel="Consumption")
for (j, i_w) in enumerate(w_indices)
    plot!(p3, model.h_grid, c_star[:, i_w], label=labels[j], linewidth=2)
end

display(plot(p1, p2, p3, layout=(1,3), size=(1200,400)))

             
#SIMULATION 

function simulate_economy(model, e_policy, T=1100, burn_in=100)
    @unpack w_grid, h_grid, P, h_step, δ, n_h, func_f = model
    
    #storage
    h_path = zeros(T)
    w_path = zeros(T)
    e_path = zeros(T)
    c_path = zeros(T)
    y_path = zeros(T)

    #inital condintion
    h_idx = findfirst(x -> x >= 1.0, h_grid)
    w_idx = 4 #middle wage

    #simulation loop
    Random.seed!(1234)

    for t in 1:T
        h_val = h_grid[h_idx]
        w_val = w_grid[w_idx]

        #policy
        e_val = e_policy[h_idx, w_idx]

        #storing in storage
        h_path[t] = h_val
        w_path[t] = w_val
        e_path[t] = e_val
        c_path[t] = w_val * func_f(h_val) * (1 - e_val)
        y_path[t] = c_path[t]

        #h
        h_next_val = h_val + e_val - δ
        h_idx_next = round(Int, h_next_val / h_step) + 1
        h_idx = clamp(h_idx_next, 1, n_h)

        #stochastic transition w 

        r = rand()
        cumsum_p = cumsum(P[w_idx, :])
        w_idx = findfirst(x -> x >= r, cumsum_p)
    end

    #deleting burn in 
    valid_range = (burn_in + 1):T
    return h_path[valid_range], w_path[valid_range], e_path[valid_range], c_path[valid_range], y_path[valid_range]
end

#running simulation 
h_sim, w_sim, e_sim, c_sim, y_sim = simulate_economy(model, e_star)

#PLOT OF PATHS
t_plot = 1:100
p_sim1 = plot(t_plot, h_sim[t_plot], title="Human Capital", legend=false)
p_sim2 = plot(t_plot, w_sim[t_plot], title="Wage Shock", legend=false)
p_sim3 = plot(t_plot, e_sim[t_plot], title="Education Effort", legend=false)
p_sim4 = plot(t_plot, c_sim[t_plot], title="Consumption", legend=false)

display(plot(p_sim1, p_sim2, p_sim3, p_sim4, layout=(2,2)))

#CORRELATIONS 
cor_y_e = cor(y_sim, e_sim)
cor_w_e = cor(w_sim, e_sim)

println("Correlation (Earnings, Education): $cor_y_e")
println("Correlation (Wage Shock, Education): $cor_w_e")