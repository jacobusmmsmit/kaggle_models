library(tidyverse)
library(ggplot2)

avocado <- read_csv("avocado.csv")

avocado %>%
    mutate(year = as.factor(year)) %>%
    group_by(region, year) %>%
    summarise(avg = mean(AveragePrice)) %>%
    ggplot(aes(x=year, y=avg)) +
    geom_violin(trim = T) +
    geom_jitter(shape = 16,position=position_jitter(0.05))

avocado %>%
    transmute(region = as.factor(region), AveragePrice=AveragePrice, Date=Date, year=year) %>%
    group_by(region, year) %>%
    ggplot(aes(x=reorder(region, AveragePrice, FUN = median),y=AveragePrice)) +
    geom_boxplot(outlier.color = "black", outlier.shape = 16, outlier.size = 1, notch = FALSE)
