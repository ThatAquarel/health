library(ComplexHeatmap)

setwd("C:/Users/xia_t/Desktop/Projects/health")
categorized_factors <- read.csv("./prediction/results/ordered_factors_2003_2022_high_countries.csv")

drop <- c("X", "Category", "Series.Name", "Series.Code")
country_attributions= categorized_factors[!(names(categorized_factors) %in% drop)]

