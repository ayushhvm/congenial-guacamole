from django.core.management.base import BaseCommand
import numpy as np
from sklearn.preprocessing import normalize
from attendance.utils.face_recognition import CosineKNNClassifier

class Command(BaseCommand):
    help = 'Verifies the improvement of k-NN over 1-NN using synthetic data'

    def handle(self, *args, **options):
        self.stdout.write("--- Verifying k-NN Improvement ---")
        
        # 1. Generate Synthetic Data
        # 3 Classes, with some overlap/noise
        np.random.seed(42)
        n_features = 512
        n_samples_per_class = 20
        
        # Centers for 3 classes
        centers = normalize(np.random.randn(3, n_features))
        
        X_train = []
        y_train = []
        
        for i in range(3):
            # Generate samples around center
            # Noise level 0.2 creates reasonable spread
            samples = centers[i] + np.random.randn(n_samples_per_class, n_features) * 0.2
            X_train.append(normalize(samples))
            y_train.extend([i] * n_samples_per_class)
            
        X_train = np.vstack(X_train)
        y_train = np.array(y_train)
        
        # Add a few outliers/mislabeled-like points to training data to confuse 1-NN
        # e.g. a point from class 0 but very close to center of class 1
        outlier = centers[1] + np.random.randn(1, n_features) * 0.05
        X_train = np.vstack([X_train, normalize(outlier)])
        y_train = np.append(y_train, 0) # Labelled as 0, but is actually near class 1 center
        
        # 2. Generate Test Data (noisy queries near class centers)
        X_test = []
        y_test = []
        n_test = 50
        for i in range(3):
            samples = centers[i] + np.random.randn(n_test, n_features) * 0.25 # slightly more noise
            X_test.append(normalize(samples))
            y_test.extend([i] * n_test)
            
        X_test = np.vstack(X_test)
        y_test = np.array(y_test)
        
        # 3. Train & Evaluate 1-NN (Simulated by k=1)
        clf_1nn = CosineKNNClassifier(k=1)
        clf_1nn.fit(X_train, y_train)
        preds_1nn = clf_1nn.predict(X_test)
        acc_1nn = np.mean(preds_1nn == y_test)
        
        # 4. Train & Evaluate k-NN (k=5)
        clf_knn = CosineKNNClassifier(k=5)
        clf_knn.fit(X_train, y_train)
        preds_knn = clf_knn.predict(X_test)
        acc_knn = np.mean(preds_knn == y_test)
        
        self.stdout.write(f"1-NN Accuracy (Simulation): {acc_1nn:.4f}")
        self.stdout.write(f"k-NN (k=5) Accuracy:        {acc_knn:.4f}")
        
        if acc_knn > acc_1nn:
            self.stdout.write(self.style.SUCCESS(f"Improvement: +{(acc_knn - acc_1nn)*100:.2f}%"))
        elif acc_knn == acc_1nn:
            self.stdout.write(self.style.WARNING("No change in accuracy (data might be too clean or too separated)"))
        else:
            self.stdout.write(self.style.ERROR("Performance degraded!"))
            
        # Test specific outlier case
        # Query near the outlier (which is Class 0 labeled, but geometrically in Class 1)
        # 1-NN should predict Class 0 (wrongly trusting the outlier)
        # k-NN should predict Class 1 (trusting the majority neighbors which are Class 1)
        
        query_outlier = normalize(outlier + np.random.randn(1, n_features) * 0.01)
        
        p1 = clf_1nn.predict(query_outlier)[0]
        p5 = clf_knn.predict(query_outlier)[0]
        
        self.stdout.write("\n--- Outlier Robustness Test ---")
        self.stdout.write(f"Query point is geometrically in Class 1 cluster, but near a training outlier labeled Class 0.")
        self.stdout.write(f"1-NN Prediction: Class {p1} (Likely 0 -> Wrong/Overfit)")
        self.stdout.write(f"k-NN Prediction: Class {p5} (Likely 1 -> Correct/Robust)")
