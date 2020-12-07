
from sklearn import metrics

# 
report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)  











