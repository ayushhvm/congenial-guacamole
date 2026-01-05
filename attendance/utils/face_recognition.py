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
    Simple Nearest Neighbor classifier using Cosine Similarity.
    Stores normalized embeddings and finds the one with highest dot product.
    """
    def __init__(self):
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
        For compatibility with sklearn API style
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        X_norm = normalize(X)
        # Dot product: (n_query, n_features) @ (n_features, n_stored) -> (n_query, n_stored)
        similarities = np.dot(X_norm, self.X.T)
        best_indices = np.argmax(similarities, axis=1)
        return self.y[best_indices]
        
    def predict_score(self, x_single):
        """
        Predict (label, score) for a single sample.
        """
        x_norm = normalize(x_single.reshape(1, -1))
        similarities = np.dot(self.X, x_norm.T).flatten()
        best_idx = np.argmax(similarities)
        return self.y[best_idx], similarities[best_idx]


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
    
    def process_image(self, image_path):
        """
        Single-pass processing: Detect -> Align -> Embed.
        Returns the Best Face (largest) as (embedding, aligned_crop).
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
            for face in faces:
                bbox = face.bbox.astype(int) # x1, y1, x2, y2
                # Convert to x, y, w, h format for compatibility
                x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
                bbox_fmt = (x, y, w, h)
                
                embedding = face.embedding
                if embedding is None:
                    recognitions.append((None, 0.0, bbox_fmt, "No embedding"))
                    continue
                
                name, confidence = self.predict(embedding, threshold)
                
                if name is None:
                    recognitions.append((None, confidence, bbox_fmt, "Confidence too low"))
                else:
                    recognitions.append((name, confidence, bbox_fmt, "Success"))
            
            return recognitions
            
        except Exception as e:
            print(f"Error recognizing faces from {image_path}: {str(e)}")
            return []

    def load_faces_from_directory(self, directory):
        """
        Load all faces from directory structure
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
                
                # Use process_image to get embedding directly
                embedding, _ = self.process_image(img_path)
                
                if embedding is not None:
                    X.append(embedding) # Store EMBEDDINGS not IMAGES
                    Y.append(person_name)
                    faces_loaded += 1
            
            print(f"Loaded {faces_loaded} faces for {person_name}")
        
        # NOTE: This now returns EMBEDDINGS in X, not images.
        return np.asarray(X), np.asarray(Y)
    
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
    
    def train_model(self, X_embeddings, Y_labels, save_path=None):
        """
        Train Classifier on face embeddings
        """
        # Encode labels
        self.encoder = LabelEncoder()
        Y_encoded = self.encoder.fit_transform(Y_labels)
        
        # Train Cosine Classifier
        self.model = CosineKNNClassifier()
        self.model.fit(X_embeddings, Y_encoded)
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
        
        # Calculate accuracy
        predictions = self.model.predict(X_embeddings)
        accuracy = np.mean(predictions == Y_encoded)
        print(f"Training accuracy: {accuracy * 100:.2f}%")
        
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
