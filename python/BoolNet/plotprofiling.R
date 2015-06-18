library(reshape2)
library(ggplot2)

data <- fromJSON('[ { "name": "old", "test file": "test/speed_test_net.json", "truth table": 0.3408808084903285, "weighted linear lsb": 0.004942284896969795, "worst sample linear msb": 0.004880278185009956, "move then revert": 0.00016988369170576335, "weighted linear msb": 0.0049167584860697385, "hierarchical linear lsb": 0.00485384501516819, "simple": 0.004858120996505022, "set mask": 2.1720095537602903e-05, "worst sample linear lsb": 0.004838823713362217, "hierarchical linear msb": 0.004882042715325952, "error per output": 0.004918594192713499 }, { "name": "pure python", "move then revert": 2.8168599965283647e-05, "test file": "test/speed_test_net.json", "hierarchical linear lsb": 0.7002369207999436, "error per output": 0.7212269398998614, "truth table": 40.16241245759993, "worst sample linear lsb": 0.7252092540999001, "hierarchical linear msb": 0.7002681008001673, "set mask": 4.009460026281886e-05, "simple": 0.6657383476998803, "weighted linear lsb": 0.6691028901997924, "weighted linear msb": 0.6678816670002561, "worst sample linear msb": 0.712391266800114 } ]')

data$"test file" <- NULL

plot.data <- melt(data, id.vars="name")
ggplot(plot.data) + geom_boxplot(aes(x=variable, y=value, color=name))

vals <- data[,c(2,3,4,5,6,7,8,9,10,11,12)]
div <- vals[2,] / vals[1,]
div <- melt(div)

ggplot(div) + geom_point(aes(x=variable, y=value))
