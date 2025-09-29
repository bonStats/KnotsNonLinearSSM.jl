library(ggplot2)
library(readr)
library(dplyr)
library(tidyr)
library(latex2exp)
library(patchwork)
library(gt)
library(matrixStats)

results5 <- read_csv("github/KnotsNonLinearSSM.jl/example/results/ssm-test-d5.csv")


gg_outer <- results |> pivot_longer(cols = everything()) |>
 ggplot() + geom_boxplot(aes(x = name, y = value)) +
  theme_bw() +
  xlab("Particle Filter") +
  ylab(TeX("$\\log\\hat{\\gamma}^N_n(1)$"))


gg_inner <- results |> pivot_longer(cols = everything()) |> filter(value > -250) |>
  ggplot() + geom_boxplot(aes(x = name, y = value)) +
  theme_bw() +
  xlab("") +
  ylab("")

gg_outer + inset_element(gg_inner, left = 0.55, bottom = 0.25, right = 0.95, top = 0.75)



results5 <- read_csv("github/KnotsNonLinearSSM.jl/example/results/ssm-test-d5.csv") |>
  mutate(dim = 5)
results4 <- read_csv("github/KnotsNonLinearSSM.jl/example/results/ssm-test-d4.csv") |>
  mutate(dim = 4)
results3 <- read_csv("github/KnotsNonLinearSSM.jl/example/results/ssm-test-d3.csv") |>
  mutate(dim = 3)
results2 <- read_csv("github/KnotsNonLinearSSM.jl/example/results/ssm-test-d2.csv") |>
  mutate(dim = 2)
results1 <- read_csv("github/KnotsNonLinearSSM.jl/example/results/ssm-test-d1.csv") |>
  mutate(dim = 1)

results <- bind_rows(results1,results2,results3,results4,results5) |> 
  pivot_longer(cols = -dim) |> 
  group_by(dim,name)

 
results |> 
  summarise(log_mean = logSumExp(value) - log(n())) |>
  pivot_wider(names_from = dim, values_from = log_mean) |>
  gt() |> fmt_number(decimals = 2,use_seps = FALSE) |>
  as_latex() |>
  cat()
  
  

results |> 
  summarise(sd_log = sd(value)) |>
  pivot_wider(names_from = dim, values_from = sd_log) |>
  gt() |> fmt_number(decimals = 2,use_seps = FALSE) |>
  as_latex() |>
  cat()


