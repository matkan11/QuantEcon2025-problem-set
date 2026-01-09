using LinearAlgebra, Statistics, Plots

#1 parameters

Z_vals = [1, 2, 3]
X_vals = 0:5 
n_z = length(Z_vals)
n_x = length(X_vals)
n_total = n_x * n_z #18

#Matrix P (transition for Z)
P = [0.6 0.3 0.1;
     0.2 0.6 0.2;
     0.1 0.3 0.6]

#Policy function X_t+1
function policy_sigma(x, z_idx)
    if z_idx == 1       #z1
        return 0
    elseif z_idx == 2   #z2
        return x
    elseif z_idx == 3   #z3, two possibilities
        if x <= 4
            return min(x + 1, 5)
        else #(if X = 5)
            return 3
        end
    end
end

#Making transition matrix Q
Q = zeros(n_total, n_total)

#helper function
get_index(x, z_idx) = (x * n_z) + z_idx

for x_now in X_vals
    for z_now in Z_vals
        
        #current row
        row_idx = get_index(x_now, z_now)
        
        #finding next X
        x_next = policy_sigma(x_now, z_now)
        
        #looping over possible next Z
        for z_next in Z_vals
            #Transition 
            prob = P[z_now, z_next]
            col_idx = get_index(x_next, z_next)
            
            #Add to matrix
            Q[row_idx, col_idx] = prob
        end
    end
end

println("Transition Matrix Q with size: ", size(Q))
Q

#2. computing stationary distribution
#iterative method

psi_0 = fill(1.0/n_total, 1, n_total)
psi_star = psi_0 * (Q^2000) 
psi_vec = vec(psi_star)

#marginal distributions and means
marginal_x = zeros(n_x)
marginal_z = zeros(n_z)
cond_mean_num = zeros(n_z)

for x_idx in 1:n_x
    x_val = X_vals[x_idx]
    for z_idx in 1:n_z
        
       #probability at (x,z)
        flat_idx = get_index(x_val, z_idx)
        prob = psi_vec[flat_idx]
        
        #marginal X, sum over all Z
        marginal_x[x_idx] += prob
        
        #marginal Z, sum over all X
        marginal_z[z_idx] += prob
        
        #conditional mean of x given z
        cond_mean_num[z_idx] += x_val * prob
    end
end

mean_x = sum(X_vals .* marginal_x)
cond_mean_x = cond_mean_num ./ marginal_z

println("Marginal X: ", round.(marginal_x, digits=4))
println("Marginal Z: ", round.(marginal_z, digits=4))
println("Mean X: ", round(mean_x, digits=4))
println("Conditional Means E[X|Z]: ", round.(cond_mean_x, digits=4))

#Plots

#Marginal Distribution of X
p1 = bar(X_vals, marginal_x, title="Marginal of X", legend=false, xlabel="X", ylabel="Prob")

#Marginal Distribution of Z
p2 = bar(["z1", "z2", "z3"], marginal_z, title="Marginal of Z", legend=false, xlabel="Z", color=:orange)

#Conditional Mean E[X | Z]
p3 = bar(["z1", "z2", "z3"], cond_mean_x, title="Conditional Mean E[X | Z]", legend=false, xlabel="Z", ylabel="Value", color=:green)