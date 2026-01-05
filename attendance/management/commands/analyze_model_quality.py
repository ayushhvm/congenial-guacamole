from django.core.management.base import BaseCommand
from attendance.models import FaceEmbedding, Student
import numpy as np
from sklearn.preprocessing import normalize
from collections import defaultdict

class Command(BaseCommand):
    help = 'Analyzes the quality of face embeddings and suggests optimal thresholds'

    def handle(self, *args, **options):
        self.stdout.write("Fetching embeddings...")
        embeddings_qs = FaceEmbedding.objects.select_related('student').all()
        
        if not embeddings_qs.exists():
            self.stdout.write(self.style.ERROR("No embeddings found in database."))
            return

        # 1. Organize Data
        student_embeddings = defaultdict(list)
        all_embeddings = []
        all_labels = []
        
        for emb in embeddings_qs:
            try:
                arr = emb.get_embedding()
                if arr.shape == (512,):
                    # Normalized on load just in case, though they should be raw from InsightFace
                    # InsightFace usually returns normalized 512d, but good to ensure
                    arr = normalize(arr.reshape(1, -1)).flatten()
                    student_embeddings[emb.student.student_id].append(arr)
                    all_embeddings.append(arr)
                    all_labels.append(emb.student.student_id)
            except Exception:
                pass

        students = list(student_embeddings.keys())
        self.stdout.write(f"Loaded {len(all_embeddings)} embeddings for {len(students)} students.")

        # 2. Compute Centroids
        centroids = {}
        for s_id, embs in student_embeddings.items():
            # Mean direction vector
            mean_vec = np.mean(embs, axis=0)
            # Re-normalize centroid to lie on hypersphere
            centroids[s_id] = normalize(mean_vec.reshape(1, -1)).flatten()

        # 3. Analyze Intra-Class (Consistency)
        # How close are a student's images to their own centroid?
        intra_scores = []
        outliers = []
        
        for s_id, embs in student_embeddings.items():
            centroid = centroids[s_id]
            for i, emb in enumerate(embs):
                score = np.dot(emb, centroid)
                intra_scores.append(score)
                
                # Flag potential outliers (arbitrary low score relative to self)
                if score < 0.6: # If similarity to own average is < 0.6, it's suspicious
                   outliers.append(f"{s_id} Image #{i+1} (Score: {score:.4f})")

        avg_intra = np.mean(intra_scores)
        min_intra = np.min(intra_scores)
        
        self.stdout.write(self.style.SUCCESS(f"\n--- Consistency (Intra-Class) ---"))
        self.stdout.write(f"Average Similarity to Self: {avg_intra:.4f}")
        self.stdout.write(f"Worst Similarity to Self: {min_intra:.4f}")
        if outliers:
            self.stdout.write(self.style.WARNING(f"Potential Outliers (bad images): {len(outliers)} found"))
            for o in outliers[:5]:
                self.stdout.write(f"  - {o}")
            if len(outliers) > 5: self.stdout.write(f"  ... and {len(outliers)-5} more.")

        # 4. Analyze Inter-Class (Separability)
        # How close are different students to each other?
        inter_scores = []
        lookalikes = []
        
        centroid_matrix = np.array([centroids[s] for s in students])
        # Dot product of all centroids against all centroids
        sim_matrix = np.dot(centroid_matrix, centroid_matrix.T)
        
        # We only care about upper triangle, excluding diagonal
        for i in range(len(students)):
            for j in range(i + 1, len(students)):
                score = sim_matrix[i, j]
                inter_scores.append(score)
                
                if score > 0.5: # Hard checking for very close people
                    lookalikes.append((score, students[i], students[j]))

        avg_inter = np.mean(inter_scores)
        max_inter = np.max(inter_scores)

        self.stdout.write(self.style.SUCCESS(f"\n--- Separability (Inter-Class) ---"))
        self.stdout.write(f"Average Similarity b/w Others: {avg_inter:.4f}")
        self.stdout.write(f"Highest Similarity b/w Others: {max_inter:.4f}") # Worst case for false positive
        
        lookalikes.sort(reverse=True)
        if lookalikes:
            self.stdout.write(self.style.WARNING(f"Closest Lookalikes (Potential False Positives):"))
            for score, s1, s2 in lookalikes[:5]:
                self.stdout.write(f"  - {s1} <-> {s2}: {score:.4f}")

        # 5. Suggest Threshold
        # A good threshold is between the worst intra (min valid score) and best inter (max impostor score)
        # Ideally Max_Inter < Threshold < Min_Intra
        
        self.stdout.write(self.style.SUCCESS(f"\n--- Recommendations ---"))
        
        if max_inter < min_intra:
            ideal_threshold = (max_inter + min_intra) / 2
            self.stdout.write(f"PERFECT SEPARATION POSSIBLE.")
            self.stdout.write(f"Recommended Threshold: {ideal_threshold:.4f}")
            self.stdout.write(f"(Range: {max_inter:.4f} < T < {min_intra:.4f})")
        else:
            self.stdout.write(self.style.WARNING(f"OVERLAP DETECTED."))
            self.stdout.write(f"Max Inter ({max_inter:.4f}) is higher than Min Intra ({min_intra:.4f}).")
            self.stdout.write("Some valid faces are less similar to their average than some lookalikes are to each other.")
            
            # Suggest conservative threshold (prioritize low False Positives)
            suggested = max(max_inter + 0.05, 0.4)
            self.stdout.write(f"Suggested Conservative Threshold: {suggested:.4f} (may reject some valid faces)")

