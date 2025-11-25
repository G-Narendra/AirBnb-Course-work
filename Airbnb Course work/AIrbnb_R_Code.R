library(dplyr)
library(stringr)
library(caret)
library(pROC)
library(rpart) 
library(rpart.plot)
library(xgboost)
library(ggplot2)
df=read.csv("C:/Users/naren/OneDrive/Desktop/KD/Coursework/airbnb data/listings/listings - final.csv",header=TRUE)
View(df)
dim(df)
str(df)
colSums(is.na(df))
df$host_since=as.Date(df$host_since,format="%d-%m-%Y")
df$last_review=as.Date(df$last_review,format="%d-%m-%Y")
df$price=as.numeric(gsub("[\\$,]","",df$price))
df$host_response_rate[df$host_response_rate == "N/A"]=NA
df$host_response_rate=as.numeric(gsub("%","",df$host_response_rate))/100
df$host_acceptance_rate[df$host_acceptance_rate == "N/A"]=NA
df$host_acceptance_rate=as.numeric(gsub("%","",df$host_acceptance_rate))/100
df$bathrooms_num_raw=as.numeric(sub("([0-9.]+).*","\\1",df$bathrooms_text))
df$host_is_superhost_flag=as.numeric(df$host_is_superhost == "t")
df$instant_bookable_flag =as.numeric(df$instant_bookable == "t")
cols_for_target_check=c("review_scores_rating","reviews_per_month")
df=df[!apply(is.na(df[,cols_for_target_check]),1,all),]
df=df[!is.na(df$host_acceptance_rate),]
################################### Outlier Capping#########################################
cap_by_iqr=function(x) {
  if (!is.numeric(x)) x=suppressWarnings(as.numeric(as.character(x)))
  if (all(is.na(x))) return(x)
  rng=range(x,na.rm=TRUE)
  if (rng[1] == rng[2]) return(x)
  Q1=quantile(x,0.25,na.rm=TRUE)
  Q3=quantile(x,0.75,na.rm=TRUE)
  IQR_val=Q3 - Q1
  lower_bound=Q1 - 1.5 * IQR_val
  upper_bound=Q3 + 1.5 * IQR_val
  x=ifelse(!is.na(x) & x < lower_bound,lower_bound,x)
  x=ifelse(!is.na(x) & x > upper_bound,upper_bound,x)
  return(x)
}
cols_to_cap <- c("host_listings_count","bedrooms","beds","minimum_nights","maximum_nights",
  "minimum_minimum_nights","maximum_minimum_nights","minimum_maximum_nights","maximum_maximum_nights",
  "minimum_nights_avg_ntm","maximum_nights_avg_ntm","number_of_reviews",
  "calculated_host_listings_count_entire_homes","calculated_host_listings_count_private_rooms","bathrooms_num_raw")
df <- df %>% mutate(across(all_of(cols_to_cap), cap_by_iqr))
############################################ NA Value Imputation ######################
get_mode=function(x,na.rm=FALSE) {
  if (na.rm) {x=x[!is.na(x)]}
  freq=table(x)
  mode_val=names(freq)[freq == max(freq)]
  if (is.numeric(x)) { mode_val=as.numeric(mode_val)}
  return(mode_val[1])}
impute_mode=function(x) { x[is.na(x)]=get_mode(x,na.rm=TRUE); x}
impute_median=function(x) { x[is.na(x)]=median(x,na.rm=TRUE); x }
impute_mean=function(x) { x[is.na(x)]=mean(x,na.rm=TRUE); x}
normalize= function(x) { rng=max(x,na.rm=TRUE) - min(x,na.rm=TRUE); if (rng == 0) rep(NA,length(x)) else (x - min(x,na.rm=TRUE))/rng }
cols_for_target_check=c("review_scores_rating","reviews_per_month")
df=df[!apply(is.na(df[,cols_for_target_check]),1,all),]
df=df[!is.na(df$host_acceptance_rate),]
cols_to_impute_median=c("minimum_minimum_nights","maximum_minimum_nights","minimum_maximum_nights","maximum_maximum_nights",
                        "minimum_nights_avg_ntm","maximum_nights_avg_ntm")
cols_to_impute_mode=c("bathrooms_num_raw","beds","host_response_rate")
cols_to_impute_mean=c("review_scores_rating","review_scores_cleanliness","review_scores_value","review_scores_accuracy",
                      "review_scores_checkin","review_scores_location","review_scores_communication")
for (col in cols_to_impute_median) {df[[col]]=impute_median(df[[col]])}
for (col in cols_to_impute_mean) {df[[col]]=impute_mean(df[[col]])}
for (col in cols_to_impute_mode) {df[[col]]=impute_mode(df[[col]])}
################################### Bedrooms Filling #######################################
df$bathrooms_scaled=normalize(df$bathrooms_num_raw)
df$price_scaled=normalize(df$price)
df$accommodates_scaled=normalize(df$accommodates)
df$beds_scaled=normalize(df$beds)
df_complete=df[!is.na(df$bedrooms),]
df_missing =df[is.na(df$bedrooms),]
features=c("bathrooms_scaled","price_scaled","accommodates_scaled","beds_scaled")
X_train=as.matrix(df_complete[features])
y_train=df_complete$bedrooms
X_test =as.matrix(df_complete[features])
y_test =df_complete$bedrooms
set.seed(123) 
model=xgboost(data=X_train,label=y_train,nrounds=4000,objective="reg:squarederror",verbose=0)
pred_test=predict(model,X_test)
mse=mean((y_test - pred_test)^2)
r2 =1 - sum((y_test - pred_test)^2)/sum((y_test - mean(y_test))^2)
cat("Test MSE (XGBoost Bedrooms):",mse,"\n")
cat("Test R² (XGBoost Bedrooms):",r2,"\n")
X_missing=as.matrix(df_missing[features])
pred_missing=predict(model,X_missing)
df$bedrooms[is.na(df$bedrooms)]=round(pred_missing)
################################ Formula ###################################################
df$value_density=df$price/df$accommodates
df$avg_review_score=rowMeans(df[,c("review_scores_rating","review_scores_cleanliness","review_scores_value",
                                   "review_scores_value","review_scores_location","review_scores_communication",
                                   "review_scores_checkin","review_scores_accuracy")],na.rm=TRUE)
df$booking_rate_proxy=(365 - df$availability_365)/365
df$norm_booking_rate=normalize(df$booking_rate_proxy)
df$norm_long_demand=normalize(log(df$reviews_per_month + 1))
df$norm_value_density=normalize(log(df$value_density+1))
df$price_competitiveness=1 - df$norm_value_density
df$demand_value_score=rowMeans(df[,c("norm_booking_rate","norm_long_demand","instant_bookable_flag","price_competitiveness")],na.rm=TRUE)
df$norm_guest_satisfaction=normalize(df$avg_review_score/5.0)
df$norm_response=normalize(df$host_response_rate)
df$norm_acceptance=normalize(df$host_acceptance_rate)
df$quality_host_score=(df$norm_guest_satisfaction + df$norm_response + df$norm_acceptance + df$host_is_superhost_flag)/4
df$success_score=(0.5 * df$demand_value_score) + (0.5 * df$quality_host_score)
success_threshold=quantile(df$success_score,0.75,na.rm=TRUE)
df$listing_success=factor( ifelse(df$success_score >= success_threshold,"Good","Bad"),levels=c("Bad","Good"))
################################ Prepare Remaining Data ####################################
df$is_shared_bath=as.numeric(str_detect(df$bathrooms_text,"shared|Shared"))
df$num_bathrooms=df$bathrooms_num_raw
top_amenities=c("Wifi","Kitchen","Essentials","Heating","Washer","Dryer","Air conditioning","Smoke alarm","Carbon monoxide alarm","Free parking")
for (amenity in top_amenities) { df[[paste0("has_",tolower(gsub(" ","_",amenity)))]]=as.numeric(str_detect(df$amenities,regex(amenity,ignore_case=TRUE))) }
df$verification_count=str_count(df$host_verifications,",") + 1
df$verification_count[df$host_verifications == "[]"]=0
ref_date=as.Date("2023-09-06")
df$host_tenure_days=as.numeric(ref_date - df$host_since)
df$review_recency_days=as.numeric(ref_date - df$last_review)
leakage_and_intermediate_cols=c(
  "host_total_listings_count",
  "host_response_rate","host_acceptance_rate","host_is_superhost","accommodates","price","availability_365","instant_bookable","reviews_per_month",
  "review_scores_rating","review_scores_accuracy","review_scores_cleanliness","review_scores_checkin","review_scores_communication","review_scores_location",
  "review_scores_value","success_score","demand_value_score","quality_host_score","value_density","avg_review_score","norm_booking_rate","norm_long_demand",
  "norm_value_density","price_competitiveness","norm_guest_satisfaction","norm_response","norm_acceptance","host_is_superhost_flag","instant_bookable_flag",
  "bathrooms_scaled","price_scaled","accommodates_scaled","beds_scaled","host_since","first_review","last_review","amenities","bathrooms_text",
  "host_verifications","number_of_reviews_ltm","number_of_reviews_l30d","bathrooms_num_raw","booking_rate_proxy")
df$room_type=ifelse(df$room_type == "Entire home/apt","Entire home",df$room_type)
df$room_type=as.factor(df$room_type)
df_final_model=df %>%
  select(-intersect(leakage_and_intermediate_cols,names(df))) %>%
  filter(!is.na(host_listings_count)) %>%
  mutate(
    has_availability=as.factor(has_availability),
    room_type=as.factor(room_type),
    neighbourhood_cleansed=as.factor(neighbourhood_cleansed))
####################################### Hypotheses Making ###################################
tab <- table(df$verification_count, df$listing_success)
prop_tab <- prop.table(tab, margin = 1)
df_summary <- as.data.frame(prop_tab)
colnames(df_summary) <- c("verification_count", "listing_success", "proportion")
ggplot(df_summary, aes(x = as.numeric(as.character(verification_count)),y = proportion,color = listing_success,
                       group = listing_success)) +geom_line(size = 1.2) +geom_point(size = 3) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Proportion of Good vs Bad by Verification Count",x = "Verification Count",y = "Proportion") +theme_minimal()
tab <- table(df$room_type, df$listing_success)
prop_good <- prop.table(tab, margin = 1)[, "Good"]
df_summary <- as.data.frame(prop.table(tab, margin = 1))
colnames(df_summary) <- c("room_type", "listing_success", "proportion")
ggplot(df_summary, aes(x = room_type, y = proportion, fill = listing_success)) +
  geom_bar(stat = "identity", position = "dodge") +scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Proportion of Good vs Bad Listings by Room Type",x = "Room Type",y = "Proportion") +theme_minimal()
######################################## Modelling #########################################
set.seed(42)
trainIndex=createDataPartition(df_final_model$listing_success,p=0.8,list=FALSE)
train_dirty=df_final_model[trainIndex,]
test_final =df_final_model[-trainIndex,]
df_minority=subset(train_dirty,listing_success == "Good")
df_majority=subset(train_dirty,listing_success == "Bad")
sample_size=nrow(df_minority)
df_majority_sample=df_majority[sample(nrow(df_majority),sample_size),]
training_data_balanced=rbind(df_minority,df_majority_sample)
training_data_balanced$listing_success=factor(training_data_balanced$listing_success,levels=c("Bad","Good"))
test_final$listing_success=factor(test_final$listing_success,levels=c("Bad","Good"))
print(table(training_data_balanced$listing_success))
print(table(test_final$listing_success))
#################################### Logistic regression ###################################
model_formula_all=listing_success ~ .
glm_model_balanced=glm(model_formula_all,data =training_data_balanced,family="binomial")
glm_pred_prob_balanced=predict(glm_model_balanced,newdata=test_final,type   ="response")
thresholds=seq(0.2,0.95,by=0.05)
results=data.frame(Threshold=thresholds,Accuracy=NA,Sensitivity=NA,Specificity=NA,F1=NA)
for (i in seq_along(thresholds)) {
  t=thresholds[i]
  pred_class=factor(ifelse(glm_pred_prob_balanced > t,"Good","Bad"),levels=c("Bad","Good"))
  cm=confusionMatrix(pred_class,test_final$listing_success)
  results$Accuracy[i]   =cm$overall["Accuracy"]
  results$Sensitivity[i]=cm$byClass["Sensitivity"]
  results$Specificity[i]=cm$byClass["Specificity"]
  precision=cm$byClass["Pos Pred Value"]
  recall   =cm$byClass["Sensitivity"]
  results$F1[i]=ifelse(is.na(precision) | is.na(recall) | (precision+recall)==0,NA,2 * precision * recall/(precision + recall))}
print(results)
best_by_f1=results[which.max(results$F1),]
best_by_acc=results[which.max(results$Accuracy),]
print(best_by_f1)
print(best_by_acc)
summary(glm_model_balanced)
################################### Decision Tree#########################################
trControl=trainControl(method="cv",number=5,classProbs=TRUE,summaryFunction=twoClassSummary)
tree_model_final=train(model_formula_all,data=training_data_balanced,method="rpart",
                       trControl=trControl,tuneLength=20,metric="ROC",control=rpart.control(minsplit=20,minbucket=7,maxdepth=12))
cat("Optimal cp chosen:",tree_model_final$bestTune$cp,"\n")
tree_pred_prob=predict(tree_model_final,newdata=test_final,type="prob")[,"Good"]
roc_curve=roc(test_final$listing_success,tree_pred_prob,levels=c("Bad","Good"))
best_threshold_coords=coords(roc_curve,"best",ret="threshold",best.method="youden")
optimal_threshold=best_threshold_coords[1,1]
cat("Optimal Probability Threshold:",optimal_threshold,"\n")
tree_pred_class_optimized=factor(ifelse(tree_pred_prob > optimal_threshold,"Good","Bad"),levels=c("Bad","Good"))
final_conf_matrix=confusionMatrix(tree_pred_class_optimized,test_final$listing_success)
cat("Decision Tree Results","\n")
print(final_conf_matrix)
cat("Tree AUC-ROC:",round(roc_curve$auc,4),"\n")
tree_pruned_viz=prune(tree_model_final$finalModel,cp=0.005) 
rpart.plot(tree_pruned_viz,type=2,extra=104,under=TRUE,faclen=0,cex=0.7)