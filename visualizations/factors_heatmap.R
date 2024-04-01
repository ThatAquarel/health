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

# categorize countries
categorized_countries <- read.csv("./prediction/results/predicted_categories.csv")
categories <- c("Low", "Medium-low", "Medium", "Medium-high", "High")
countries_categories <- categories[categorized_countries[["Predicted.Category"]] + 1]

# color function
library(circlize)
col_fun_heatmap <- colorRamp2(c(-0.2, 0, 0.05), c("blue", "white", "red"))

# ABR risk level annotation (columns)

abr_ha <- HeatmapAnnotation(
  ABR_risk = factor(countries_categories, levels=categories),
  annotation_legend_param = list(
    ABR_risk = list(ABR_risk_color_bar = "discrete"))
)

# draw heat map
Heatmap(
  mat_country_attributions,
  name="factor_importance",

  #row_split=factors_categories,
  column_split=factor(countries_categories, levels=categories),
  column_title=NULL,
  cluster_column_slices=FALSE,
  border = TRUE,
  
  top_annotation = abr_ha,
  heatmap_legend_param = list(ABR_risk_color_bar = "discrete"),

  col=col_fun_heatmap
)
