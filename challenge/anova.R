library(multcomp)
library("report")

challenge_flat$Province <- as.factor(challenge_flat$Province)

res_aov <- aov(
  Emissions ~ Province,
  data = challenge_flat
)

report(res_aov)

posthoc <- glht(res_aov, linfct = mcp(Province = "Tukey"))
summary(posthoc, test = adjusted("bonferroni"))
