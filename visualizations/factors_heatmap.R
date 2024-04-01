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
  ABR_risk_penicillin = countries_categories,
  annotation_legend_param = list(
    ABR_risk_penicillin = list(
      title="ABR_risk_penicillin (DDD/1,000/day)",
      labels=c(
        "(0.67,  5.76] - Low",
        "(5.76, 10.81] - Medium-low",
        "(10.81, 15.87] - Medium",
        "(15.87, 20.93] - Medium-high",
        "(20.93, 25.98] - High"
      ),
      ABR_risk_color_bar = "discrete"
    )
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

# draw heat map
Heatmap(
  mat_country_attributions,
  name="factor_importance",
  
  column_split=countries_categories,
  column_title=NULL,
  cluster_column_slices=FALSE,
  column_names_rot = 45,
  top_annotation = abr_ha,
  
  #row_split=factors_categories,
  right_annotation = factor_ha,
  row_dend_reorder = TRUE,
  border=TRUE,

  col=col_fun_heatmap
)
