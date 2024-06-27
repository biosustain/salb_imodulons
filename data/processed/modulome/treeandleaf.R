

#-- Load packages
#-- Load packages
library("TreeAndLeaf")
library("RedeR")
library("igraph")
library("ape")
library("ggtree")
library("dendextend")
library("dplyr")
library("ggplot2")
library("RColorBrewer")

# Read the data
data <- read.csv('treeleaf_test.csv', row.names = 1)
head(data)

hc <- hclust(dist(data))
hc

den <- as.dendrogram(hc)
den

clus <- cutree(hc, 2)
g <- split(names(clus), clus)

p <- ggtree(hc, linetype='dashed')
clades <- sapply(g, function(n) MRCA(p, n))

p <- groupClade(p, clades, group_name='subtree') + aes(color=subtree)

d <- data.frame(label = names(clus), 
                cyl = data[names(clus), "salb_Phosphate"])

p %<+% d + 
  layout_dendrogram()

#-- Convert the 'hclust' object into a 'tree-and-leaf' object
tal <- treeAndLeaf(hc)

#--- Call RedeR application
rdp <- RedPort()
calld(rdp)
resetd(rdp)

#--- Send the tree-and-leaf to the interactive R/Java interface
addGraph(obj = rdp, g = tal, gzoom=75)

