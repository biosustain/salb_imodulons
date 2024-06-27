library(UpSetR)


top5 <- read.csv("Orthogroups/Orthogroups_binary.tsv", header = TRUE, sep = "\t")

# Assuming your dataset contains columns '1FgAgo1_A', '1FgAgo2_A', ..., '8FoAgo4_A'
all_sets <- colnames(top5)[2:ncol(top5)]  # Exclude the first column (sequences)

# Open a PDF device with specified width and height
pdf("orthogroups_upset_plot.pdf", width = 10, height = 5)

# Specify the order of the sets
set_order = c("Sac", "Bsu", "Pae", "Sen", "Eco", "Mtu", "Sal")

# Create an UpSet plot with the specified set order
upset(top5, sets = set_order, sets.bar.color = "#56B4E9", order.by = "freq", keep.order = T)

# Close the PDF device
dev.off()