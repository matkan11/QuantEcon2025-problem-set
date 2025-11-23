using Statistics, Plots, DelimitedFiles, Distributions, LinearAlgebra,IterativeSolvers
# Task 1: Load the dataset from the CVS file 
data = readdlm("problem_sets//PS1//asset_returns.csv", ',',Float64; skipstart=1)
(T, n) = size(data)

# Task 2: compute the sample mean vector and sample covariance matrix of the data 
μ = transpose(mean(data, dims=1))
Σ = cov(data)

println("Size of mean vector μ: ", size(μ))
println("Size of covariance matrix Σ: ", size(Σ))
# Task 3: Form the (n+2)x(n+2) Matrix System Ax = b with target return 0.10.

#set target return
μ_bar = 0.10

#creating the vector of ones - because all weights must sum to 1 as per constraint
ones_n = ones(n)

#creating the zero vector of size n
zeros_n = zeros(n)

#creating the matrix A as per the equation in the problem statement
A = vcat(
    hcat(Σ, μ, ones_n),
    hcat(transpose(μ), 0.0, 0.0),
    hcat(transpose(ones_n), 0.0, 0.0)
)

#creating the vector b as per the equation in the problem statement
b = vcat(zeros_n, μ_bar, 1.0)

#checking if correct: (n+2)x(n+2) matrix, n+2
println("Size of A matrix: ", size(A))
println("Size of b vector: ", size(b))

#reporting the condition number of A
cond_A = cond(A)
println("Condition number of A: ", cond_A)

# Task 4: part a - solve using backslash operator + task 5 
x_backslash = A\b

#Total computational time for backslash operator:
@time  x_backslash = A\b;

#Calculating the residual norm for backslash operator:
b_norm = norm(b)
residual_norm_backslash = norm(A*x_backslash - b)
relative_residual_backslash = residual_norm_backslash / b_norm

println("No iterations needed for backslash operator.")
println("Relative residual norm using backslash operator: ", relative_residual_backslash)
println("Solution x using backslash operator (first 10 elements): ", x_backslash[1:10])
#Task 4: part b solve using normal equations + task 5

#forming the new system 
AtA =  transpose(A) * A
Atb = transpose(A) * b

println("Size of AtA matrix: ", size(AtA))
println("Size of Atb vector: ", size(Atb))

#Checking for zeros on the diagonal
diag_AtA = diag(AtA)
has_zeros_diagonal = any(diag_AtA .== 0.0)
println("Does AtA have zeros on the diagonal? ", has_zeros_diagonal)

#Calculating condition number of AtA
cond_AtA = cond(AtA)
println("Condition number of AtA: ", cond_AtA)

#check diagonal dominance: |a_ii| >= sum_{j != i} |a_ij| for all i
is_diagonally_dominant = true
for i in 1:size(AtA, 1)
    diagonal_element = abs(AtA[i, i])
    off_diagonal_sum = sum(abs.(AtA[i, :]))- diagonal_element

    if diagonal_element <= off_diagonal_sum
        is_diagonally_dominant = false
        break
    end
end
println("Is AtA diagonally dominant? ", is_diagonally_dominant)
println("Matrix AtA is not diagonally dominant, so Gauss-Seidel may not converge.")


# Task 4 part c - Conjugate Gradient method + task 5

#check for symmetry and positive definiteness
is_symmetric = issymmetric(AtA)
is_pos_def = isposdef(AtA)
println("Is AtA symmetric? ", is_symmetric)
println("Is AtA positive definite? ", is_pos_def)

if is_symmetric && is_pos_def
    
    #solving using Conjugate Gradient method
    x_cg, history_cg = cg(AtA, Atb, log=true)

    #Total computational time for Conjugate Gradient method:
    @time x_cg, history_cg = cg(AtA, Atb, log=true);
    #Calculating the residual norm for Conjugate Gradient method:
    residual_norm_cg = norm(A*x_cg - b)
    relative_residual_cg = residual_norm_cg / b_norm

    println("Number of iterations for Conjugate Gradient method: ", length(history_cg.data))
    println("Relative residual norm using Conjugate Gradient method: ", relative_residual_cg)
    println("Computational time for Conjugate Gradient method recorded above.")

else
    println("AtA is not symmetric positive definite; Conjugate Gradient method cannot be applied.")
end

#Task 4 part d - using GMRES + task 5
x_gmres, history_gmres = gmres(A, b, log=true)
#Total computational time for GMRES:
@time x_gmres, history_gmres = gmres(A, b, log=true);

#Calculating the residual norm for GMRES:
residual_norm_gmres = norm(A*x_gmres - b)  
relative_residual_gmres = residual_norm_gmres / b_norm

println("Number of iterations for GMRES: ", length(history_gmres.data))
println("Relative residual norm using GMRES: ", relative_residual_gmres)
println("Computational time for GMRES recorded above.")

#Task 4 part e - Preconditioned GMRES + task 5

#Creating given matrix P
diag_S = diag(Σ)
P_diag = vcat(diag_S, 1.0, 1.0)
P = Diagonal(P_diag)

#we need to solve P^{-1}Ax = P^{-1}b
P_inv = inv(P)
x_pgmres, history_pgmres = gmres(P_inv * A, P_inv * b, log=true)

#Total computational time for Preconditioned GMRES:
@time x_pgmres, history_pgmres = gmres(P_inv * A, P_inv * b, log=true);

#Calculating the residual norm for Preconditioned GMRES:
residual_norm_pgmres = norm(A*x_pgmres - b)
relative_residual_pgmres = residual_norm_pgmres / b_norm

println("Number of iterations for Preconditioned GMRES: ", length(history_pgmres.data))
println("Relative residual norm using Preconditioned GMRES: ", relative_residual_pgmres)
println("Computational time for Preconditioned GMRES recorded above.")


#Task 6 - report optimal portfolio weights and report portfolio variance \sigma_p^2 = w^T Σ w, veirfy is sums to 1 and meets expected return

#For backslash operator
weights_backslash = x_backslash[1:n]

#verify that weights sum to 1
sum_weights_backslash = sum(weights_backslash)

#verify if expected return is met
return_backslash = transpose(μ) * weights_backslash

#portfolio variance
portfolio_variance_backsklash = transpose(weights_backslash) * Σ * weights_backslash
println("Backslash Operator:")
println("Sum of weights: ", sum_weights_backslash)
println("Expected return: ", return_backslash)
println("Portfolio Variance: ", portfolio_variance_backsklash)

#For Conjugate Gradient method
weights_cg = x_cg[1:n]

#verify that weights sum to 1
sum_weights_cg = sum(weights_cg)

#verify if expected return is met
return_cg = transpose(μ) * weights_cg

#portfolio variance
portfolio_variance_cg = transpose(weights_cg) * Σ * weights_cg

println("Conjugate Gradient Method:")
println("Sum of weights: ", sum_weights_cg) 
println("Expected return: ", return_cg)
println("Portfolio Variance: ", portfolio_variance_cg)

#For GMRES
weights_gmres = x_gmres[1:n]

#verify that weights sum to 1
sum_weights_gmres = sum(weights_gmres)

#verify if expected return is met
return_gmres = transpose(μ) * weights_gmres

#portfolio variance
portfolio_variance_gmres = transpose(weights_gmres) * Σ * weights_gmres

println("GMRES:")
println("Sum of weights: ", sum_weights_gmres)
println("Expected return: ", return_gmres)
println("Portfolio Variance: ", portfolio_variance_gmres)

#For Preconditioned GMRES
weights_pgmres = x_pgmres[1:n]

#verify that weights sum to 1
sum_weights_pgmres = sum(weights_pgmres)

#verify if expected return is met
return_pgmres = transpose(μ) * weights_pgmres

#portfolio variance
portfolio_variance_pgmres = transpose(weights_pgmres) * Σ * weights_pgmres

println("Preconditioned GMRES:")
println("Sum of weights: ", sum_weights_pgmres)
println("Expected return: ", return_pgmres)
println("Portfolio Variance: ", portfolio_variance_pgmres)


# Asset priving models solving many similar systems tests 
#Chosen method - Preconditioned backslash operator

#defining 50 target returns μ 
μ_targets = range(0.01, 0.10, length=50)

portfolio_std_devs = []
portfolio_returns = []

initial_guess = zeros(n + 2)

for μ_bar_loop in μ_targets

    #b vector for new target return
    b_loop = vcat(zeros(n), μ_bar_loop, 1.0)

    #solving using Preconditioned GMRES with initial guess, x0 = initial_guess
    b_loop = vcat(zeros(n), μ_bar_loop, 1.0)

    x_new = A\b_loop

    #extracting weights
    weights_new = x_new[1:n]

    #calculating portfolio variance and standard deviation
    var_p = transpose(weights_new) * Σ * weights_new
    std_dev_p = sqrt(var_p)

    #storing results for plots 
    push!(portfolio_std_devs, std_dev_p)
    push!(portfolio_returns, μ_bar_loop)
end

#Plotting the efficient frontier
plot(
    portfolio_std_devs,
    portfolio_returns,
    title = "Efficient Frontier",
    xlabel = "Portfolio Standard Deviation (Risk)",
    ylabel = "Portfolio Expected Return",
    legend = false,
    markershape = :circle,
    markerstrokewidth = 0,
    markercolor = :blue,
    markersize = 3
)
