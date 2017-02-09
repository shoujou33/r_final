# 讀取資料
titanic <- read.csv("https://storage.googleapis.com/r_rookies/kaggle_titanic_train.csv")
View(titanic)
str(titanic)
summary(titanic)

# 因分類需要使用到 "rpart"，須先install.packages("rpart")
library(ggplot2)
library(plotly)
library(rpart)
library(dplyr)
library(magrittr)

# 性別
#ggbar.sex <- ggplot(titanic, aes(x = Sex, y = Survived, fill = Sex)) + geom_bar(stat = "identity")
#plotly.sex <- ggplotly(ggbar.sex)
#plotly.sex

# Pclass
#ggplot_bar_pclass <- ggplot(titanic, aes(x = factor(Pclass), y = Survived, fill = factor(Pclass))) + geom_bar(stat = "identity", width = .7)
#ggplot_bar_pclass_plotly <- ggplotly(ggplot_bar_pclass)
#ggplot_bar_pclass_plotly

# 將資料空缺填補成眾數
summary(titanic$Embarked) # 眾數為S，且有兩個遺漏值。
titanic$Embarked <- as.character(titanic$Embarked)
class(titanic$Embarked)
titanic$Embarked[titanic$Embarked == ""] <- "S"
titanic$Embarked <- as.factor(titanic$Embarked)
class(titanic$Embarked)
table(titanic$Embarked)

# 填補遺漏值
summary(titanic$Age)
# 找出三種艙等的平均年齡
titanic$Pclass <- factor(titanic$Pclass)
ggplot(titanic, aes(x = Pclass, y = Age)) +
  geom_boxplot()
summarise(group_by(titanic, Pclass), mean_age = round(mean(Age, na.rm = TRUE)))
# 尋找填補位置
filter_1 <- is.na(titanic$Age) & titanic$Pclass == 1
filter_2 <- is.na(titanic$Age) & titanic$Pclass == 2
filter_3 <- is.na(titanic$Age) & titanic$Pclass == 3
# 填補
titanic[filter_1, ]$Age <- 38
titanic[filter_2, ]$Age <- 30
titanic[filter_3, ]$Age <- 25
summary(titanic$Age)

# 調整資料型別
titanic$Survived <- factor(titanic$Survived)
class(titanic$Survived)
n <- nrow(titanic)
# 將樣本洗牌
set.seed(32)
shuffled_titanic <- titanic[sample(n), ]
head(shuffled_titanic)

# 使用 75/25 比例分割為訓練(train)樣本/測試(test)樣本
train_ind <- 1:round(0.75 * n)
train <- shuffled_titanic[train_ind, ]
test_ind <- (round(0.75 * n) + 1):n
test <- shuffled_titanic[test_ind, ]

# 建立分類器
library(randomForest)
set.seed(32)
forest_fit <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train, ntree = 200)
# 計算分類精確度
prediction <- predict(forest_fit, newdata = test[, c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")])
head(prediction)
confusion_matrix <- table(test$Survived, prediction)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
accuracy

summary(titanic)



# 載入待預測資料
to_predict <- read.csv("https://storage.googleapis.com/py_ds_basic/kaggle_titanic_test.csv")
# 觀察待預測資料
View(to_predict)
str(to_predict)
summary(to_predict) # Age 有 86 個遺漏值、Fare 有 1 個遺漏值。

# 找出三種艙等的平均年齡
to_predict$Pclass <- factor(to_predict$Pclass)
ggplot(to_predict, aes(x = Pclass, y = Age)) +
  geom_boxplot()
summarise(group_by(to_predict, Pclass), mean_age = round(mean(Age, na.rm = TRUE)))
# 尋找填補位置
filter_1 <- is.na(to_predict$Age) & to_predict$Pclass == 1
filter_2 <- is.na(to_predict$Age) & to_predict$Pclass == 2
filter_3 <- is.na(to_predict$Age) & to_predict$Pclass == 3
# 填補遺漏值
to_predict[filter_1, ]$Age <- 41
to_predict[filter_2, ]$Age <- 29
to_predict[filter_3, ]$Age <- 24
summary(to_predict$Age)

# 尋找 Fare 平均值
fare_mean <- mean(to_predict$Fare, na.rm = TRUE)
# 填補遺漏值
to_predict$Fare[is.na(to_predict$Fare)] <- fare_mean
summary(to_predict$Fare)
View(to_predict)

# 預測資料
summary(to_predict)
to_predict$Pclass <- as.integer(to_predict$Pclass)
predicted <- predict(forest_fit, newdata = to_predict[, c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")])
to_submit <- data.frame(to_predict[, "PassengerId"], predicted)
names(to_submit) <- c("PassengerId", "Survived")
# 檢視預測資料
head(to_submit, n = 10)
# 輸出預測資料
write.csv(to_submit, file = "to_submit.csv", row.names = FALSE)