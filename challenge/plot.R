challenge_flat$Province <- as.factor(challenge$Province)

p<-ggplot(challenge_flat, aes(x=Province, y=Emissions)) + 
  geom_dotplot(binaxis='y', stackdir='center', dotsize=0.75) + geom_boxplot() + scale_x_discrete(guide = guide_axis(n.dodge = 2))

p
