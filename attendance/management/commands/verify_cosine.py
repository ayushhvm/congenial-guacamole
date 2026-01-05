from django.core.management.base import BaseCommand
import numpy as np
from sklearn.preprocessing import normalize
from attendance.utils.face_recognition import CosineKNNClassifier

class Command(BaseCommand):
    help = 'Verifies that Cosine Similarity scores do not drop with more users'

    def handle(self, *args, **options):
        self.stdout.write("Testing Cosine KNN Score vs Number of Registered People")
        self.stdout.write("-" * 60)
        self.stdout.write(f"{'Num People':<15} | {'Cosine Score (same input)':<25}")
        self.stdout.write("-" * 60)
        
        # Base data for class 0
        np.random.seed(42)
        center0 = normalize(np.random.randn(1, 512))[0]
        # 10 samples for class 0, slightly varied
        X0 = normalize(center0 + np.random.randn(10, 512) * 0.1) 
        y0 = np.zeros(10)
        
        # Test sample (very close to center0)
        # We ensure it's not identical to any training sample but very close
        test_sample = normalize((center0 + np.random.randn(512) * 0.05).reshape(1, -1))
        
        # Test with varying number of TOTAL classes (1 user + N others)
        counts = [2, 5, 10, 20, 50, 100, 200, 500]
        
        for n_total in counts:
            n_others = n_total - 1
            
            # Generate other classes
            X_others = []
            y_others = []
            for i in range(1, n_total):
                center = normalize(np.random.randn(1, 512)) # Random center
                samples = normalize(center + np.random.randn(10, 512) * 0.1)
                X_others.append(samples)
                y_others.append(np.full(10, i))
                
            if X_others:
                X = np.vstack([X0, np.vstack(X_others)])
                y = np.concatenate([y0, np.concatenate(y_others)])
            else:
                X = X0
                y = y0
                
            # Train CosineKNN
            clf = CosineKNNClassifier()
            clf.fit(X, y)
            
            # Predict
            label, score = clf.predict_score(test_sample)
            
            # Check if it still predicts class 0 and what the score is
            is_correct = (label == 0)
            self.stdout.write(f"{n_total:<15} | {score:.4f} {'(Correct)' if is_correct else '(Wrong)'}")
