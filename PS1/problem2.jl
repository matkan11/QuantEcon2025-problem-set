using LinearAlgebra, Printf, PrettyTables
#2.1
function system_solve(α, β)
    return ones(5)
end
# The exact solution is [1, 1, 1, 1, 1]' regardless of alpha and beta.
#2.2
function solutions(α, β)
A = [ 1 -1 0 α-β β; 0 1 -1 0 0; 0 0 1 -1 0; 0 0 0 1 -1; 0 0 0 0 1] 
B = [α, 0, 0, 0, 1]
exact_solution = system_solve(α, β)
solution_backslash = A\B 
relative_residual = norm(A*solution_backslash - B)/norm(B)
condition_number = cond(A)
    return exact_solution, solution_backslash, relative_residual, condition_number
end
#2.3
α = 0.1
β_values = [1.0, 10.0, 100.0]
append!(β_values, [10.0^p for p in 3:12])

results = []
for β in β_values
    exact_solution, solution_backslash, relative_residual, condition_number = solutions(α, β)
    x1_exact = exact_solution[1]
    x1_computed = solution_backslash[1]
    push!(results, (β, x1_exact, x1_computed, condition_number, relative_residual))
end
pretty_table(results;
    header = (["β", "Exact x₁", "Computed x₁", "Condition Number", "Relative Residual"],
              ["", "", "", "", ""]),
    formatters = ft_printf("%1.2e", 5),
    header_crayon = crayon"yellow bold",
    display_size = (-1, -1))
    # Findings:
    # As β increases, the condition number of the matrix A also increases significantly, 
    # indicating that the system becomes more ill-conditioned.
    # This leads to larger relative residuals, suggesting that the computed 
    # solutions deviate more from the exact solution as β grows.
    # The computed value of x₁ also diverges from the exact value of 1 as β increases, 
    # highlighting the impact of ill-conditioning on solution accuracy.  