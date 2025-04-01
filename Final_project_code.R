# Load necessary libraries
library(shiny)
library(tidyverse)
library(caret)
library(randomForest)
library(class)
library(e1071)
library(ggplot2)
library(plotly)
library(rsample)
library(recipes)
library(parsnip)
library(shinydashboard)
library(workflows)
library(kknn)
library(yardstick)
library(nnet)  # For Neural Network
library(cluster)  # For Silhouette Score

# Read the dataset
df <- read.csv("E:/Downloads/5th Sem/INT234/house_prices_df.csv")  # Choose the file


# Data Preprocessing
df_clean <- df %>%
  select(SalePrice, OverallCond, GrLivArea, GarageCars, BsmtFinSF1, TotalBsmtSF,
         X1stFlrSF, LotArea, GarageArea, FullBath, YearBuilt) %>%
  mutate(OverallCond = as.factor(OverallCond))

# Split the data into training and testing sets (80% train, 20% test)
set.seed(345)
split <- initial_split(df_clean, prop = 0.80)
train_data <- training(split)
test_data <- testing(split)

# Define the recipe for preprocessing (dummy encoding for categorical variables)
recipe <- recipe(SalePrice ~ ., data = train_data) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

# Define models
linear_reg_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

knn_model <- nearest_neighbor(neighbors = 5) %>%
  set_engine("kknn") %>%
  set_mode("regression")

# Neural Network Model
nn_model <- mlp(hidden_units = 5) %>%
  set_engine("nnet")  %>%
set_mode("regression")

# Prepare workflow function for training each model
train_model <- function(model, train_data, recipe) {
  workflow() %>%
    add_model(model) %>%
    add_recipe(recipe) %>%
    fit(data = train_data)
}

# UI
ui <- dashboardPage(
  dashboardHeader(title = "House Prediction"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Linear Regression", tabName = "lr", icon = icon("chart-line")),
      menuItem("KNN", tabName = "knn", icon = icon("chart-bar")),
      menuItem("Neural Network", tabName = "nn", icon = icon("network-wired")),
      menuItem("Unsupervised Learning - KMeans", tabName = "kmeans", icon = icon("cogs")),
      menuItem("Model Comparison", tabName = "comparison", icon = icon("th-list"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(tabName = "lr", 
              fluidRow(
                box(title = "Linear Regression Model Accuracy", status = "primary", solidHeader = TRUE, width = 12, 
                    h3(strong(textOutput("lr_accuracy"))),
                    p("This chart compares the predicted house prices against the actual house prices. The red line represents perfect predictions where predicted values equal actual values."),
                    plotlyOutput("lr_plot", height = "600px"),
                    h4(strong("Summary Notes")),
                   # p("Linear Regression models the relationship between variables by fitting a linear equation. It works well for datasets where the relationship between input variables and the output variable is linear.")
                    p("Linear Regression uses scatter plots with a regression line to depict the relationship between independent and dependent variables. The chart explains how well the line fits the data, highlighting trends or deviations. For example, in sales prediction, the line shows how sales increase with advertising spend.")
                )
              )),
      tabItem(tabName = "knn", 
              fluidRow(
                box(title = "KNN Model Accuracy", status = "primary", solidHeader = TRUE, width = 12, 
                    h3(strong(textOutput("knn_accuracy"))),
                    p("This chart compares the predicted house prices against the actual house prices. The red line represents perfect predictions where predicted values equal actual values."),
                    plotlyOutput("knn_plot", height = "600px"),
                    h4(strong("Summary Notes")),
                   # p("K-Nearest Neighbors (KNN) is a non-parametric method used for regression. It predicts the target value based on the average of the nearest neighbors.")
                    p("K-Nearest Neighbors (KNN) typically uses scatter plots with decision boundaries to illustrate classification. Points are grouped based on their proximity to the K-nearest neighbors. The chart describes how well data points are separated into classes, emphasizing local decision-making.")
                )
              )),
      tabItem(tabName = "nn", 
              fluidRow(
                box(title = "Neural Network Model Accuracy", status = "primary", solidHeader = TRUE, width = 12, 
                    h3(strong(textOutput("nn_accuracy"))),
                    p("This chart compares the predicted house prices against the actual house prices. The red line represents perfect predictions where predicted values equal actual values."),
                    plotlyOutput("nn_plot", height = "600px"),
                    h4(strong("Summary Notes")),
                   # p("Neural Networks are a type of machine learning algorithm inspired by the structure of the brain. It is particularly useful for complex datasets where relationships are non-linear. This model uses multiple layers of nodes (neurons) to capture intricate patterns in the data.")
                    p("Neural Networks often use confusion matrices or boundary plots. For classification tasks, boundary plots illustrate regions assigned to different classes. Confusion matrices display performance metrics like accuracy, showing misclassified points. These charts describe how effectively the model handles non-linear relationships.")
                    )
              )),
      tabItem(tabName = "kmeans", 
              fluidRow(
                box(title = "KMeans Clustering", status = "primary", solidHeader = TRUE, width = 12, 
                    h3("KMeans Clustering Results"),
                    p("This chart shows the results of the KMeans clustering, where we cluster houses based on features like GrLivArea and GarageArea."),
                    plotlyOutput("kmeans_plot", height = "600px"),
                    h4(strong("Summary Notes")),
                   p("K-Means clustering is visualized through scatter plots showing clusters with distinct colors. Each cluster is centered around a calculated mean, and the chart demonstrates how data is grouped based on similarity, helping to identify patterns or group behaviors.")
                    # p("KMeans is an unsupervised learning algorithm that groups data points into clusters based on similarity. It works well for discovering inherent groupings in the data."),
                    h4(strong("KMeans Clustering Accuracy:")),
                    p(textOutput("kmeans_accuracy"))
                )
              )),
      tabItem(tabName = "comparison",
              fluidRow(
                box(title = "Model Accuracy Comparison", status = "primary", solidHeader = TRUE, width = 12, 
                    p("This bar chart compares the accuracies of the models. The accuracy is calculated as 1 minus the average relative error."),
                    plotlyOutput("comparison_plot", height = "600px"))
                    p("A comparison of machine learning algorithms—such as Linear Regression, K-Nearest Neighbors (KNN), K-Means Clustering, and Neural Networks—reveals distinct strengths and weaknesses based on their underlying methodologies and how they approach data problems. Linear Regression is a statistical model ideal for predicting continuous outcomes based on a linear relationship between variables, making it simple and interpretable. However, its performance deteriorates when faced with non-linear relationships or outliers. In contrast, KNN is a flexible, instance-based learning algorithm that assigns class labels based on proximity to neighboring data points. It works well with smaller datasets and is intuitive, but it can be computationally expensive and sensitive to the choice of 'K' and the distance metric used, especially in high-dimensional spaces. K-Means Clustering, an unsupervised learning algorithm, excels at grouping similar data points into clusters, making it useful for exploratory analysis and segmentation tasks. However, it requires the user to specify the number of clusters in advance and can struggle with non-spherical clusters or datasets with varying densities. Neural Networks, particularly deep learning models, are highly effective for complex tasks like image recognition or speech processing due to their ability to capture non-linear relationships. However, they require large datasets, significant computational resources, and can be challenging to interpret, often considered as "black-box" models. Each algorithm has its specific use case, and the choice of which to use depends on factors such as the type of data, problem complexity, interpretability needs, and available computational resources.")
              ))
    )
  )
)

# Server
server <- function(input, output, session) {
  
  # Train and predict for Linear Regression
  lr_fit <- train_model(linear_reg_model, train_data, recipe)
  lr_preds <- predict(lr_fit, new_data = test_data)$.pred
  lr_accuracy <- 1 - mean(abs(lr_preds - test_data$SalePrice) / test_data$SalePrice)
  
  output$lr_accuracy <- renderText({
    paste("Linear Regression Accuracy: ", round(lr_accuracy * 100, 2), "%")
  })
  
  output$lr_plot <- renderPlotly({
    plot <- ggplot(data.frame(Predicted = lr_preds, Actual = test_data$SalePrice), 
                   aes(x = Actual, y = Predicted)) +
      geom_point() +
      geom_abline(slope = 1, intercept = 0, color = "red") +
      ggtitle("Linear Regression Prediction vs Actual") +
      theme_minimal()
    ggplotly(plot)  # Convert ggplot to interactive plotly plot
  })
  
  # Train and predict for KNN
  knn_fit <- train_model(knn_model, train_data, recipe)
  knn_preds <- predict(knn_fit, new_data = test_data)$.pred
  knn_accuracy <- 1 - mean(abs(knn_preds - test_data$SalePrice) / test_data$SalePrice)
  
  output$knn_accuracy <- renderText({
    paste("KNN Accuracy: ", round(knn_accuracy * 100, 2), "%")
  })
  
  output$knn_plot <- renderPlotly({
    plot <- ggplot(data.frame(Predicted = knn_preds, Actual = test_data$SalePrice), 
                   aes(x = Actual, y = Predicted)) +
      geom_point() +
      geom_abline(slope = 1, intercept = 0, color = "red") +
      ggtitle("KNN Prediction vs Actual") +
      theme_minimal()
    ggplotly(plot)  # Convert ggplot to interactive plotly plot
  })
  
  # Train and predict for Neural Network
  nn_fit <- train_model(nn_model, train_data, recipe)
  nn_preds <- predict(nn_fit, new_data = test_data)$.pred
  nn_accuracy <- 1 - mean(abs(nn_preds - test_data$SalePrice) / test_data$SalePrice)
  
  output$nn_accuracy <- renderText({
    paste("Neural Network Accuracy: ", round(nn_accuracy * 100, 2), "%")
  })
  
  output$nn_plot <- renderPlotly({
    plot <- ggplot(data.frame(Predicted = nn_preds, Actual = test_data$SalePrice), 
                   aes(x = Actual, y = Predicted)) +
      geom_point() +
      geom_abline(slope = 1, intercept = 0, color = "red") +
      ggtitle("Neural Network Prediction vs Actual") +
      theme_minimal()
    ggplotly(plot)  # Convert ggplot to interactive plotly plot
  })
  
  # KMeans Clustering
  kmeans_model <- kmeans(select(train_data, GrLivArea, GarageArea), centers = 3)
  kmeans_preds <- kmeans_model$cluster
  
  output$kmeans_plot <- renderPlotly({
    plot <- ggplot(data.frame(GrLivArea = train_data$GrLivArea, 
                              GarageArea = train_data$GarageArea, 
                              Cluster = as.factor(kmeans_model$cluster)),
                   aes(x = GrLivArea, y = GarageArea, color = Cluster)) +
      geom_point() +
      ggtitle("KMeans Clustering (GrLivArea vs GarageArea)") +
      theme_minimal()
    ggplotly(plot)  # Convert ggplot to interactive plotly plot
  })
  
  # KMeans Clustering Accuracy (Silhouette Score)
  silhouette_score <- silhouette(kmeans_model$cluster, dist(select(train_data, GrLivArea, GarageArea)))
  kmeans_accuracy <- mean(silhouette_score[, 3])
  
  output$kmeans_accuracy <- renderText({
    paste("KMeans Clustering Silhouette Score: ", round(kmeans_accuracy, 2))
  })
  
  # Comparison of Model Accuracy
  model_accuracies <- data.frame(
    Model = c("Linear Regression", "KNN", "Neural Network", "KMeans"),
    Accuracy = c(lr_accuracy, knn_accuracy, nn_accuracy, kmeans_accuracy)
  )
  
  output$comparison_plot <- renderPlotly({
    comparison_plot <- ggplot(model_accuracies, aes(x = Model, y = Accuracy, fill = Model)) +
      geom_bar(stat = "identity") +
      labs(title = "Model Accuracy Comparison") +
      theme_minimal()
    ggplotly(comparison_plot)  # Convert ggplot to interactive plotly plot
  })
}

# Run the app
shinyApp(ui = ui, server = server)

