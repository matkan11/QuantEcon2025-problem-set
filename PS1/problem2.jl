using LinearAlgebra, Printf, PrettyTables
#2.1
function system_solve(α, β)
    return ones(5)
end
# The exact solution is [1, 1, 1, 1, 1]' regardless of alpha and beta.
#2.2
function solutions(α, β)
    #Define the matrix A and vector B
A = [ 1 -1 0 α-β β; 0 1 -1 0 0; 0 0 1 -1 0; 0 0 0 1 -1; 0 0 0 0 1] 

B = [α, 0, 0, 0, 1]

exact_solution = system_solve(α, β)

#solve using backslash operator
solution_backslash = A\B 

#calculate how wrong our answer is (relative residual)
relative_residual = norm(A*solution_backslash - B)/norm(B)

#calculate condition number of A (how sensitive A is to small changes)
condition_number = cond(A)

    return exact_solution, solution_backslash, relative_residual, condition_number
end

#2.3
α = 0.1

#generate beta values
β_values = [1.0, 10.0, 100.0]
append!(β_values, [10.0^p for p in 3:12])

#store results in a table
results = Matrix{Any}(undef, length(β_values), 5)

#loop through beta values and collect results
for (i, β) in enumerate(β_values)
    exact_solution, solution_backslash, relative_residual, condition_number = solutions(α, β)
    
    #extract first element of solutions for reporting
    x1_exact = exact_solution[1]
    x1_computed = solution_backslash[1]

    # Store formatted results in the matrix
    results[i, 1] = @sprintf("%.1e", β)
    results[i, 2] = @sprintf("%.6f", x1_exact)
    results[i, 3] = @sprintf("%.6f", x1_computed)
    results[i, 4] = @sprintf("%.2e", condition_number)
    results[i, 5] = @sprintf("%.2e", relative_residual)

end

#Display results in a formatted table

pretty_table(results; header = ["β", "Exact x₁", "Computed x₁", "Condition Number", "Relative Residual"],
    header_crayon = crayon"yellow bold",
    alignment = [:right, :right, :right, :right, :right,])


#Pretty table does not cooperate, same error occurs in class notes



# Findings:
# As β increases, the condition number of the matrix A also increases significantly, 
# indicating that the system becomes more ill-conditioned.
# This leads to larger relative residuals, suggesting that the computed 
# solutions deviate more from the exact solution as β grows.
# The computed value of x₁ also diverges from the exact value of 1 as β increases, 
# highlighting the impact of ill-conditioning on solution accuracy.  