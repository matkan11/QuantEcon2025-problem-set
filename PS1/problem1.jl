using Distributions, Random, StatsPlots

#from the hint 
Random.seed!(1234)  # For reproducibility

#define parameters, X poisson 
λ = 1.0
dist = Poisson(λ)

#mean and standard deviation
μ = mean(dist)
σ = std(dist)

#n sizes, and realizations
sample_sizes= [5, 25, 100, 1000]
n_realizations = 1000

#simulating and plotting
plot_list=[]

for n in sample_sizes
    sample_means = [mean(rand(dist, n)) for _ in 1:n_realizations]
    
  #standardizing mean using CLT
  standard_error = σ/sqrt(n)
  Z_n = (sample_means .- μ) ./ standard_error

  #plotting 
  p = histogram(Z_n, 
  normalize = :pdf, 
  color = :lightblue,
  alpha = 0.7,
  title = "Standardized Sample Means (n=$n)",
    xlabel = "Z_n",
    ylabel = "Density")

    #superimpose standard normal distribution
    plot!(p, -4:0.1:4, pdf.(Normal(0,1), -4:0.1:4), color = :red, lw=2, label="N(0,1) PDF")

    push!(plot_list, p)
end

#combining plots

final_plot = plot(plot_list..., layout = (2,2), size=(800,600))

display(final_plot)

#end 