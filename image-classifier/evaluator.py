from img_classifier_model_loader import ImgClassifierModelLoader
from dataset_loader import DatasetLoader
import metrics as metrics
import numpy as np

if __name__ == "__main__":
    img_classifier_model_loader = ImgClassifierModelLoader("../image-classifier/77-model/training_resnet50")
    model = img_classifier_model_loader.call()

    dataset_loader = DatasetLoader()
    _, _, test_images, test_labels = dataset_loader.call()

    y_pred = []
    for img in test_images:
        result = model.predict(img)
        y_pred.append(result)

    y_pred = np.array(y_pred)

    accuracy = metrics.accuracy_score(test_labels, y_pred)
    precision = metrics.precision_score(test_labels, y_pred)
    recall = metrics.recall_score(test_labels, y_pred)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
