import cv2 as cv
import os
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from sklearn.preprocessing import LabelEncoder, normalize
import pickle
from django.conf import settings


class CosineKNNClassifier:
    """
    k-Nearest Neighbor classifier using Cosine Similarity.
    Features:
    - Weighted voting (closer neighbors count more)
    - Consensus score (confidence)
    """
    def __init__(self, k=5):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        """
        X: (n_samples, n_features)
        y: (n_samples,)
        """
        # Normalize stored embeddings for cosine similarity
        self.X = normalize(X)
        self.y = np.array(y)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        # Backward compatibility for models trained before k-NN update
        if not hasattr(self, 'k'):
            self.k = 5

        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        X_norm = normalize(X)
        # Dot product: (n_query, n_features) @ (n_features, n_stored) -> (n_query, n_stored)
        similarities = np.dot(X_norm, self.X.T)
        
        predictions = []
        for i in range(len(X)):
            # Get top k indices
            # argsort is ascending, so take last k and reverse
            top_k_idx = np.argsort(similarities[i])[-self.k:][::-1]
            top_k_scores = similarities[i][top_k_idx]
            top_k_labels = self.y[top_k_idx]
            
            # Weighted voting
            vote_counts = {}
            for label, score in zip(top_k_labels, top_k_scores):
                if label not in vote_counts:
                    vote_counts[label] = 0.0
                # Weight can be just the score (cosine similarity)
                # You can also square it to punish lower scores more: score**2
                vote_counts[label] += score 
                
            best_label = max(vote_counts, key=vote_counts.get)
            predictions.append(best_label)
            
        return np.array(predictions)
        
    def predict_score(self, x_single):
        """
        Predict (label, confidence_score) for a single sample.
        Confidence is calculated as the weighted fraction of the winning class.
        """
        # Backward compatibility
        if not hasattr(self, 'k'):
            self.k = 5

        x_norm = normalize(x_single.reshape(1, -1))
        similarities = np.dot(self.X, x_norm.T).flatten()
        
        # Get top k
        top_k_idx = np.argsort(similarities)[-self.k:][::-1]
        top_k_scores = similarities[top_k_idx]
        top_k_labels = self.y[top_k_idx]
        
        vote_counts = {}
        total_weight = 0.0
        
        for label, score in zip(top_k_labels, top_k_scores):
            # Only consider positive similarities for meaningful voting
            weight = max(0, score) 
            if label not in vote_counts:
                vote_counts[label] = 0.0
            vote_counts[label] += weight
            total_weight += weight
            
        if total_weight == 0:
            # Should not happen unless all similarities are negative/zero
            return None, 0.0
            
        best_label = max(vote_counts, key=vote_counts.get)
        # Confidence: Proportion of total weight associated with the winner
        # Alternatively, could just return the max average score.
        # Let's return the average score of the supporting neighbors for the winner
        # as it represents "how close" they are, rather than just "how many".
        # But 'confidence' usually implies certainty.
        
        # Let's use a hybrid: (Total Weight of Winner) / k  ? 
        # No, that depends on k.
        # Let's use: (Total Weight of Winner) / (Total Weight of Top K)
        # This represents "purity" of the neighborhood.
        # AND scale it by the best individual score to represent "closeness".
        
        purity = vote_counts[best_label] / total_weight
        best_single_score = top_k_scores[0]
        
        # Combined score: Purity * BestScore? 
        # Or just use the best single score but require consensus?
        # Let's stick to the simplest effective one:
        # Returns: best_label, and the weighted average similarity of the k neighbors 
        # (treating non-class neighbors as 0? No, that punishes split neighborhoods too hard).
        
        # Let's use: Average score of the matching neighbors for the winning class.
        # This tells us "how similar are the neighbors that voted for this person?"
        
        # Actuall, standard for these is often just the top 1 score, 
        # but validated by neighbors. 
        # Let's return the Top 1 Score, but we verify if it won the vote.
        # If it didn't win the vote, we might return the winner's score.
        
        # Let's go with: Return Best Label from Voting, 
        # Score = Average Cosine Similarity of the voters for that label.
        
        winner_score_sum = vote_counts[best_label] 
        # valid voters are those with label == best_label in top k
        num_voters = np.sum(top_k_labels == best_label)
        
        avg_score = winner_score_sum / num_voters if num_voters > 0 else 0.0
        
        return best_label, avg_score


class CentroidClassifier:
    """
    Centroid-based classifier using Cosine Similarity.
    Computes the mean embedding (centroid) for each class and compares 
    query embeddings to these centroids.
    
    Pros: Very fast inference (only n_classes comparisons)
    Cons: Less robust to noisy training data
    """
    def __init__(self):
        self.centroids = None  # (n_classes, n_features)
        self.classes = None    # (n_classes,)
    
    def fit(self, X, y):
        """
        Compute centroids for each class.
        X: (n_samples, n_features)
        y: (n_samples,)
        """
        X_norm = normalize(X)
        y = np.array(y)
        
        unique_classes = np.unique(y)
        centroids = []
        
        for cls in unique_classes:
            class_samples = X_norm[y == cls]
            centroid = np.mean(class_samples, axis=0)
            # Re-normalize the centroid
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            centroids.append(centroid)
        
        self.centroids = np.array(centroids)
        self.classes = unique_classes
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        X_norm = normalize(X)
        # Dot product with centroids
        similarities = np.dot(X_norm, self.centroids.T)  # (n_query, n_classes)
        best_indices = np.argmax(similarities, axis=1)
        return self.classes[best_indices]
    
    def predict_score(self, x_single):
        """
        Predict (label, confidence_score) for a single sample.
        """
        x_norm = normalize(x_single.reshape(1, -1))
        similarities = np.dot(x_norm, self.centroids.T).flatten()
        best_idx = np.argmax(similarities)
        return self.classes[best_idx], similarities[best_idx]


# Classifier Types
CLASSIFIER_KNN = 'knn'
CLASSIFIER_CENTROID = 'centroid'

def get_classifier(classifier_type='knn', **kwargs):
    """
    Factory function to get a classifier instance.
    
    Args:
        classifier_type: 'knn' or 'centroid'
        **kwargs: Additional arguments for the classifier (e.g., k=5 for k-NN)
    
    Returns:
        Classifier instance
    """
    if classifier_type == CLASSIFIER_KNN:
        k = kwargs.get('k', 5)
        return CosineKNNClassifier(k=k)
    elif classifier_type == CLASSIFIER_CENTROID:
        return CentroidClassifier()
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}. Use 'knn' or 'centroid'.")


class FaceRecognitionSystem:
    """
    Complete face recognition system using InsightFace (ArcFace)
    for both detection and recognition.
    """
    
    def __init__(self):
        self.target_size = (160, 160)
        self.arcface_app = None
        self.model = None
        self.encoder = None
        
    def initialize_arcface(self):
        """Initialize ArcFace model for detection and embeddings"""
        if self.arcface_app is None:
            # buffalo_l includes detection (RetinaFace/SCRFD) + recognition (ArcFace) + alignment
            self.arcface_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.arcface_app.prepare(ctx_id=0, det_size=(640, 640))
        return self.arcface_app
    
    def process_image(self, image_path, min_score=0.0):
        """
        Single-pass processing: Detect -> Align -> Embed.
        Returns the Best Face (largest) as (embedding, aligned_crop).
        Filters out faces with detection score < min_score.
        """
        try:
            if self.arcface_app is None:
                self.initialize_arcface()
                
            img = cv.imread(image_path)
            if img is None:
                return None, None
                
            # InsightFace expects BGR (cv2 default), so no conversion needed for detection if using cv2.imread
            # But FaceAnalysis.get() usually handles BGR images directly.
            
            faces = self.arcface_app.get(img)
            
            if not faces:
                return None, None
                
            # Sort by det_score or bbox area. Usually det_score is good.
            # Let's pick largest area to be safe for "registering" the main subject
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
            best_face = faces[0]
            
            if best_face.det_score < min_score:
                print(f"Rejected Low Quality Face in {os.path.basename(image_path)}: Score {best_face.det_score:.3f}")
                return None, None
            
            # Embedding is already computed by .get()
            embedding = best_face.embedding
            
            # Align face (crop)
            # norm_crop returns the aligned face image
            face_img = face_align.norm_crop(img, landmark=best_face.kps)
            
            return embedding, face_img
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None, None
    
    def extract_face(self, image_path):
        """
        Legacy wrapper: Just returns the aligned face image.
        """
        _, face_img = self.process_image(image_path)
        return face_img
    
    def get_embedding(self, face_img):
        """
        Legacy wrapper: Takes an image (crop), runs detection+recog AGAIN.
        WARNING: Inefficient. Use process_image() where possible.
        """
        try:
            if self.arcface_app is None:
                self.initialize_arcface()
            
            # If input is BGR? cv2.imread is BGR.
            # face_img passed here is likely already aligned/cropped.
            faces = self.arcface_app.get(face_img)
            if faces:
                return faces[0].embedding
            return None
        except Exception as e:
            print(f"Error in legacy get_embedding: {e}")
            return None

    def recognize_all_faces_from_image(self, image_path, threshold=0.5):
        """
        Recognize all faces in an image efficiently.
        Returns list of tuples: [(name, confidence, bbox, message), ...]
        """
        try:
            if self.arcface_app is None:
                self.initialize_arcface()

            img = cv.imread(image_path)
            if img is None:
                return []
                
            faces = self.arcface_app.get(img)
            
            if not faces:
                return []
            
            recognitions = []
            
            # Temporary storage to handle duplicates in the same frame
            # Map: student_id -> (confidence, face_idx)
            best_match_for_id = {}
            temp_results = []
            
            for idx, face in enumerate(faces):
                bbox = face.bbox.astype(int) # x1, y1, x2, y2
                # Convert to x, y, w, h format for compatibility
                x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
                bbox_fmt = (x, y, w, h)
                
                embedding = face.embedding
                if embedding is None:
                    # Keep failed embeddings as they don't conflict
                    temp_results.append((None, 0.0, bbox_fmt, "No embedding"))
                    continue
                
                name, confidence = self.predict(embedding, threshold)
                
                if name is None:
                    temp_results.append((None, confidence, bbox_fmt, "Confidence too low"))
                else:
                    # Potential match found
                    # Store with index to resolve conflicts later
                    temp_results.append({'idx': idx, 'name': name, 'conf': confidence, 'bbox': bbox_fmt, 'type': 'match'})
                    
                    # Logic to ensure 1 person = 1 face per frame
                    if name in best_match_for_id:
                        prev_conf, prev_idx = best_match_for_id[name]
                        if confidence > prev_conf:
                            # Current face is better match for this person
                            best_match_for_id[name] = (confidence, idx)
                        else:
                            # Previous face was better, ignore this one for this person
                            pass
                    else:
                        # First time seeing this person in this frame
                        best_match_for_id[name] = (confidence, idx)
            
            # Finalize results
            for item in temp_results:
                if isinstance(item, tuple):
                    recognitions.append(item)
                else:
                    # It's a potential match
                    name = item['name']
                    idx = item['idx']
                    bbox = item['bbox']
                    conf = item['conf']
                    
                    best_conf, best_idx = best_match_for_id.get(name, (0, -1))
                    
                    if idx == best_idx:
                        # This matches the best face for this ID
                        recognitions.append((name, conf, bbox, "Success"))
                    else:
                        # This face was identified as 'name', but another face had higher confidence for 'name'
                        # So this face is likely NOT 'name', or is a duplicate.
                        # We should mark it as Unknown or Low Confidence to avoid duplicate attendance
                        recognitions.append((None, conf, bbox, f"Duplicate ID suppression (Best: {best_conf:.2f})"))

            return recognitions
            
        except Exception as e:
            print(f"Error recognizing faces from {image_path}: {str(e)}")
            return []

    def load_faces_from_directory(self, directory):
        """
        Load all faces from directory structure.
        Enforces strict quality check (min_score=0.65) for training data.
        """
        X = []
        Y = []
        
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
            return np.array([]), np.array([])
        
        for person_name in os.listdir(directory):
            person_path = os.path.join(directory, person_name)
            
            if not os.path.isdir(person_path):
                continue
                
            faces_loaded = 0
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                # Check original image
                # Use process_image to get embedding directly with QUALITY CHECK
                # 0.65 is reasonable for "good" faces
                try:
                    img = cv.imread(img_path)
                    if img is None:
                        continue

                    # 1. Process Original
                    # We pass the loaded image directly to avoid re-reading, but our process_image currently takes path or we need to overload it.
                    # Let's just use the existing process_image by path for consistency if possible, 
                    # OR better, since we need to flip, let's modify logic to accept image array.
                    
                    # NOTE: process_image takes image_path. Let's refactor slightly to handle in-memory image 
                    # or just manually do what process_image does here for the flip.
                    
                    # Original
                    embedding_orig, _ = self._process_image_internal(img, min_score=0.65)
                    if embedding_orig is not None:
                        X.append(embedding_orig)
                        Y.append(person_name)
                        faces_loaded += 1
                        
                    # 2. Process Flipped (Augmentation)
                    img_flipped = cv.flip(img, 1) # 1 = Horizontal flip
                    embedding_flip, _ = self._process_image_internal(img_flipped, min_score=0.65)
                    if embedding_flip is not None:
                        X.append(embedding_flip)
                        Y.append(person_name)
                        # We don't increment faces_loaded for augmentation to avoid confusion in logs, 
                        # or we can mention it. Let's count it.
                        faces_loaded += 1

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

            print(f"  > {person_name}: Processed {faces_loaded//2} files -> {faces_loaded} embeddings (100% augmentation)")
        
        # NOTE: This now returns EMBEDDINGS in X, not images.
        return np.asarray(X), np.asarray(Y)

    def _process_image_internal(self, img, min_score=0.0):
        """
        Internal helper to process an already loaded image array.
        """
        if self.arcface_app is None:
            self.initialize_arcface()
            
        faces = self.arcface_app.get(img)
        
        if not faces:
            return None, None
            
        # Sort by largest area
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        best_face = faces[0]
        
        if best_face.det_score < min_score:
            return None, None
        
        embedding = best_face.embedding
        face_img = face_align.norm_crop(img, landmark=best_face.kps)
        
        return embedding, face_img
    
    def generate_embeddings(self, faces):
        """
        Legacy: If faces is a list of images, compute embeddings.
        If faces is a list of embeddings (from updated load_faces), pass through.
        """
        if len(faces) > 0 and isinstance(faces[0], np.ndarray) and faces[0].ndim == 1 and faces[0].shape[0] == 512:
            return np.asarray(faces)
            
        # Old logic
        embeddings = []
        if self.arcface_app is None:
            self.initialize_arcface()
        
        for face in faces:
            # Here face is an image
            emb = self.get_embedding(face)
            if emb is not None:
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(512))
        return np.asarray(embeddings)
    
    def train_model(self, X_embeddings, Y_labels, save_path=None, classifier_type='knn', **classifier_kwargs):
        """
        Train Classifier on face embeddings.
        
        Args:
            X_embeddings: (n_samples, n_features) array of embeddings
            Y_labels: (n_samples,) array of labels (student IDs)
            save_path: Path to save the trained model
            classifier_type: 'knn' or 'centroid'
            **classifier_kwargs: Additional args for the classifier (e.g., k=5)
        
        Returns:
            Training accuracy
        """
        # Encode labels
        self.encoder = LabelEncoder()
        Y_encoded = self.encoder.fit_transform(Y_labels)
        
        # Get classifier using factory
        self.model = get_classifier(classifier_type, **classifier_kwargs)
        self.model.fit(X_embeddings, Y_encoded)
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
        
        # Calculate accuracy
        predictions = self.model.predict(X_embeddings)
        accuracy = np.mean(predictions == Y_encoded)
        print(f"Training accuracy ({classifier_type}): {accuracy * 100:.2f}%")
        
        return accuracy
    
    def save_model(self, model_path):
        """Save trained model and encoder"""
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        encoder_path = model_path.replace('.pkl', '_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.encoder, f)
        print(f"Model saved to {model_path}")
        print(f"Encoder saved to {encoder_path}")
    
    def load_model(self, model_path):
        """Load trained model and encoder"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            encoder_path = model_path.replace('.pkl', '_encoder.pkl')
            with open(encoder_path, 'rb') as f:
                self.encoder = pickle.load(f)
            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, embedding, threshold=0.5):
        """
        Predict identity from face embedding
        """
        if self.model is None or self.encoder is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        if hasattr(self.model, 'predict_score'):
            prediction, max_score = self.model.predict_score(embedding)
            if max_score < threshold:
                return None, max_score
            name = self.encoder.inverse_transform([prediction])[0]
            return name, max_score
        else:
            # Fallback
            return None, 0.0

    # Removed recognize_from_frame, extract_face_from_array for brevity/cleanup
    # Add them back if needed for real-time video
    def recognize_from_frame(self, frame, threshold=0.5):
        """Real-time recognition"""
        if self.arcface_app is None: self.initialize_arcface()
        
        faces = self.arcface_app.get(frame)
        if not faces:
            return None, 0, None, "No face"
            
        best_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)[0]
        embedding = best_face.embedding
        bbox = best_face.bbox.astype(int)
        x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
        
        name, conf = self.predict(embedding, threshold)
        if name:
             return name, conf, (x, y, w, h), "Success"
        return None, conf, (x, y, w, h), "Low Confidence"


def train_face_recognition_model(faces_directory, output_model_path):
    print("Initializing Face Recognition System...")
    fr_system = FaceRecognitionSystem()
    print(f"Loading faces from {faces_directory}...")
    
    # This now returns embeddings directly!
    X, Y = fr_system.load_faces_from_directory(faces_directory)
    
    if len(X) == 0:
        print("No faces found!")
        return None
    
    print(f"Loaded {len(X)} embeddings from {len(set(Y))} people")
    print("Training model...")
    accuracy = fr_system.train_model(X, Y, output_model_path)
    
    print(f"Training complete! Model saved to {output_model_path}")
    return fr_system
