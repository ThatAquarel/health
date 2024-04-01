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
category_color <- function (category, color) {
  all_categories <- c(
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
    "Economy & Growth"="white",
    "Public Sector"="white",
    "Private Sector"="white",
    "Financial Sector"="white",
    "External Debt"="white"
  )
  all_categories[category] = color

  return(all_categories)
}

factor_ha <- rowAnnotation(
  environment = anno_simple(factors_categories, col=category_color("Environment", "lawngreen")),
  climate_change = anno_simple(factors_categories, col=category_color("Climate Change", "lawngreen")),
  energy_and_mining = anno_simple(factors_categories, col=category_color("Energy & Mining", "lawngreen")),
  infrastructure = anno_simple(factors_categories, col=category_color("Infrastructure", "lawngreen")),
  agriculture_and_rural_development = anno_simple(factors_categories, col=category_color("Agriculture & Rural Development", "lawngreen")),
  urban_developement = anno_simple(factors_categories, col=category_color("Urban Development", "lawngreen")),
  
  education = anno_simple(factors_categories, col=category_color("Education", "mediumseagreen")),
  science_and_technology = anno_simple(factors_categories, col=category_color("Science & Technology", "mediumseagreen")),
  
  health = anno_simple(factors_categories, col=category_color("Health", "tomato")),
  gender = anno_simple(factors_categories, col=category_color("Gender", "tomato")),
  
  social_developement = anno_simple(factors_categories, col=category_color("Social Development", "orange")),
  poverty = anno_simple(factors_categories, col=category_color("Poverty", "orange")),
  aid_effectiveness = anno_simple(factors_categories, col=category_color("Aid Effectiveness", "orange")),
  
  economy_and_growth = anno_simple(factors_categories, col=category_color("Economy & Growth", "darkcyan")),                    
  public_sector = anno_simple(factors_categories, col=category_color("Public Sector", "darkcyan")),           
  private_sector = anno_simple(factors_categories, col=category_color("Private Sector", "darkcyan")),
  financial_sector = anno_simple(factors_categories, col=category_color("Financial Sector", "darkcyan")),
  external_debt = anno_simple(factors_categories, col=category_color("External Debt", "darkcyan"))
)

# draw heat map
Heatmap(
  mat_country_attributions,
  name="factor_importance",
  
  column_split=countries_categories,
  column_title=NULL,
  cluster_column_slices=FALSE,
  top_annotation = abr_ha,
  
  #row_split=factors_categories,
  right_annotation = factor_ha,
  border=TRUE,

  col=col_fun_heatmap
)
