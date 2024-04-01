library(circlize)
library(ComplexHeatmap)

setwd("C:/Users/xia_t/Desktop/Projects/health")
categorized_factors <- read.csv("./prediction/results/ordered_factors_2003_2022_high_countries.csv")

drop <- c("X", "Category", "Series.Name", "Series.Code")
col_filter <- !(names(categorized_factors) %in% drop)
country_attributions = categorized_factors[col_filter]

# build matrix
mat_country_attributions <- data.matrix(country_attributions)

factors <- categorized_factors[["Series.Name"]]
countries <- names(categorized_factors)[col_filter]
rownames(mat_country_attributions) = factors
colnames(mat_country_attributions) = countries

# categorize factors
factors_categories <- categorized_factors[["Category"]]

# categorize countries
categorized_countries <- read.csv("./prediction/results/predicted_categories.csv")
countries_categories <- categorized_countries[["Predicted.Category"]]

# load top factors
top_factors <- read.csv("./prediction/results/top20_balanced_factors.csv")
top_factors_names <- top_factors[["Series.Name"]]

# ABR risk level annotation (columns)
abr_col_fun <- colorRamp2(c(0,4), c("white", "orangered"))
abr_ha <- HeatmapAnnotation(
  Penicillin_ABR_risk_and_usage = anno_simple(
    countries_categories,
    col=abr_col_fun,
    border = TRUE
  )
)

# Factor category annotation (rows)
factor_ha <- rowAnnotation(
  environment = anno_simple(factors_categories, col= c(
    "Environment"="lawngreen",
    "Climate Change"="lawngreen",
    "Energy & Mining"="lawngreen",
    "Infrastructure"="lawngreen",
    "Agriculture & Rural Development"="lawngreen",
    "Urban Development"="lawngreen",
    "Education"="white",
    "Science & Technology"="white",
    "Health"="white",
    "Gender"="white",
    "Social Development"="white",
    "Poverty"="white",
    "Aid Effectiveness"="white",
    "Economy & Growth"="white",
    "Public Sector"="white",
    "Private Sector"="white",
    "Financial Sector"="white",
    "External Debt"="white"
  )),
  education = anno_simple(factors_categories, col=c(
    "Environment"="white",
    "Climate Change"="white",
    "Energy & Mining"="white",
    "Infrastructure"="white",
    "Agriculture & Rural Development"="white",
    "Urban Development"="white",
    "Education"="mediumseagreen",
    "Science & Technology"="mediumseagreen",
    "Health"="white",
    "Gender"="white",
    "Social Development"="white",
    "Poverty"="white",
    "Aid Effectiveness"="white",
    "Economy & Growth"="white",
    "Public Sector"="white",
    "Private Sector"="white",
    "Financial Sector"="white",
    "External Debt"="white"
  )),
  health = anno_simple(factors_categories, col=c(
    "Environment"="white",
    "Climate Change"="white",
    "Energy & Mining"="white",
    "Infrastructure"="white",
    "Agriculture & Rural Development"="white",
    "Urban Development"="white",
    "Education"="white",
    "Science & Technology"="white",
    "Health"="tomato",
    "Gender"="tomato",
    "Social Development"="white",
    "Poverty"="white",
    "Aid Effectiveness"="white",
    "Economy & Growth"="white",
    "Public Sector"="white",
    "Private Sector"="white",
    "Financial Sector"="white",
    "External Debt"="white"
  )),
  social_developement = anno_simple(factors_categories, col=c(
    "Environment"="white",
    "Climate Change"="white",
    "Energy & Mining"="white",
    "Infrastructure"="white",
    "Agriculture & Rural Development"="white",
    "Urban Development"="white",
    "Education"="white",
    "Science & Technology"="white",
    "Health"="white",
    "Gender"="white",
    "Social Development"="white",
    "Poverty"="orange",
    "Aid Effectiveness"="orange",
    "Economy & Growth"="white",
    "Public Sector"="white",
    "Private Sector"="white",
    "Financial Sector"="white",
    "External Debt"="white"
  )),
  economy_and_growth = anno_simple(factors_categories, col=c(
    "Environment"="white",
    "Climate Change"="white",
    "Energy & Mining"="white",
    "Infrastructure"="white",
    "Agriculture & Rural Development"="white",
    "Urban Development"="white",
    "Education"="white",
    "Science & Technology"="white",
    "Health"="white",
    "Gender"="white",
    "Social Development"="white",
    "Poverty"="white",
    "Aid Effectiveness"="white",
    "Economy & Growth"="darkcyan",
    "Public Sector"="darkcyan",
    "Private Sector"="darkcyan",
    "Financial Sector"="darkcyan",
    "External Debt"="darkcyan"
  ))
)

# important factors
important_marks <- rowAnnotation(
  top_factors = anno_mark(
    at = match(top_factors_names, factors),
    labels_gp = gpar(fontsize=8),
    labels = top_factors_names,
    extend = unit(2, "mm"),
    link_width = unit(8, "mm")
  )
)

# draw heat map
col_fun <- colorRamp2(c(-0.3, 0, 0.05), c("blue", "white", "red"))

Heatmap(
  mat_country_attributions,
  name="factor_importance",
  
  column_split=countries_categories,
  column_title=NULL,
  cluster_column_slices=FALSE,
  column_names_gp = gpar(fontsize = 0, col="white"),
  top_annotation = abr_ha,
  
  #row_split=factors_categories,
  right_annotation = factor_ha,
  row_dend_reorder = TRUE,
  row_names_gp = gpar(fontsize = 0, col="white"),
  
  border=TRUE,
  show_heatmap_legend=FALSE,
  col=col_fun
) + important_marks

# legends
heatmap_lgd = Legend(
  col_fun = col_fun,
  title = "Factor importance",
  legend_height = unit(4, "cm")
)

at = seq(0,4,1)
abr_ldg = Legend(
  at = at,
  title = "Penicillin ABR risk and usage (DDD/1,000/day)",
  legend_gp = gpar(fill = abr_col_fun(at)),
  labels=c(
    "(0.67,  5.76]   Low",
    "(5.76, 10.81]   Medium-low",
    "(10.81, 15.87]   Medium",
    "(15.87, 20.93]   Medium-high",
    "(20.93, 25.98]   High"
  ),
)

pd = packLegend(heatmap_lgd, abr_ldg)

pushViewport(viewport(width = 1.0, height = 1.0))
draw(pd, x = unit(0.8, "npc"), y = unit(0.56, "npc"))
popViewport()
