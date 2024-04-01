library(ComplexHeatmap)

setwd("C:/Users/xia_t/Desktop/Projects/health")
categorized_factors <- read.csv("./prediction/results/ordered_factors_2003_2022_high_countries.csv")

drop <- c("X", "Category", "Series.Name", "Series.Code")
col_filter <- !(names(categorized_factors) %in% drop)
country_attributions = categorized_factors[col_filter]

# build matrix
factors <- categorized_factors[["Series.Name"]]
countries <- names(categorized_factors)[col_filter]

mat_country_attributions <- data.matrix(country_attributions)
#rownames(mat_country_attributions) = factors
#colnames(mat_country_attributions) = countries

# categorize factors
factors_categories <- categorized_factors[["Category"]]

# draw heat map
Heatmap(mat_country_attributions, row_split=factors_categories)
