library(treemap)

data <- read.csv("../../data/interim/data_for_treemap.csv")

png(file = ('../../figures/panel1/treemap.png'),
    width = 850,
    height = 700)

treemap(data,
        index=c("function.","iModulon"),
        vSize="size",
        type="categorical",
        vColor='function.',
        fontsize.labels=14,
        palette='Set2',
)

dev.off()