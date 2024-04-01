library(ComplexHeatmap)

setwd("C:/Users/xia_t/Desktop/Projects/health")
categorized_factors <- read.csv("./prediction/results/ordered_factors_2003_2022_high_countries.csv")

drop <- c("X", "Category", "Series.Name", "Series.Code")
col_filter <- !(names(categorized_factors) %in% drop)
country_attributions = categorized_factors[col_filter]

# build matrix
mat_country_attributions <- data.matrix(country_attributions)

# categorize factors
factors_categories <- categorized_factors[["Category"]]

# categorize countries
categorized_countries <- read.csv("./prediction/results/predicted_categories.csv")
countries_categories <- categorized_countries[["Predicted.Category"]]

# color function
library(circlize)
col_fun_heatmap <- colorRamp2(c(-0.2, 0, 0.05), c("blue", "white", "red"))

# ABR risk level annotation (columns)
abr_ha <- HeatmapAnnotation(
  ABR_risk = countries_categories,
  annotation_legend_param = list(
    ABR_risk = list(ABR_risk_color_bar = "discrete", labels=c("Low", "Medium-low", "Medium", "Medium-high", "High"))
  )
)

# Factor category annotation (rows)


# draw heat map
Heatmap(
  mat_country_attributions,
  name="factor_importance",

  column_split=countries_categories,
  column_title=NULL,
  cluster_column_slices=FALSE,
  top_annotation = abr_ha,

  col=col_fun_heatmap
)
