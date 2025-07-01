import os
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
import pickle
import argparse

# --- configuration ---
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings_facenet.pkl"
# cosine similarity: 1 is a perfect match. A higher threshold is stricter.
COSINE_THRESHOLD = 0.8
# threshold for MTCNN face detection
MIN_CONFIDENCE = 0.95
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Running on device: {DEVICE}')

# --- models ---
# keep_all=True detects all faces in the image. Important for multi-face processing.
mtcnn = MTCNN(
    image_size=160,
    margin=0,
    keep_all=True,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    device=DEVICE,
)
# inceptionResnetV1 is pre-trained on VGGFace2
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)


def encode_known_faces(known_faces_dir, encodings_file):
    """
    Encodes all faces in the known_faces_dir and saves them to a pickle file.
    ** Supports multiple faces per image file. **
    """
    known_face_encodings = []
    known_face_names = []

    print("[INFO] Starting to encode known faces...")
    for filename in os.listdir(known_faces_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(known_faces_dir, filename)
        name = os.path.splitext(filename)[0]

        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARNING] Failed to load {image_path}, skipping.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # detect and align faces. mtcnn() returns a batch of face tensors.
        faces = mtcnn(img_rgb)
        if faces is None:
            print(f"[WARNING] No face detected in {filename}. Skipping.")
            continue

        # iterate over all detected faces in the image
        print(f"[INFO] Found {len(faces)} face(s) in {filename}.")
        for i, face_to_encode in enumerate(faces):
            with torch.no_grad():
                # Get the embedding for the single face
                embedding = resnet(face_to_encode.unsqueeze(0).to(DEVICE))
                known_face_encodings.append(embedding.cpu().numpy()[0])
                known_face_names.append(name)
                print(f"  > Encoded face {i+1}/{len(faces)} for: {name}")

    with open(encodings_file, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print(f"[INFO] Encoding complete. Saved to {encodings_file}")


def load_encodings(encodings_file):
    """
    Loads face encodings and names from a pickle file.
    """
    if not os.path.exists(encodings_file):
        return [], []
    with open(encodings_file, 'rb') as f:
        return pickle.load(f)


def recognize_faces(frame, known_face_encodings, known_face_names):
    """
    Recognizes faces in a single video frame using Cosine Similarity.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # use mtcnn.detect for faster processing, getting boxes and probabilities
    boxes, probs = mtcnn.detect(img_rgb)
    names = []

    if boxes is not None:
        # filter out weak detections based on confidence
        confident_indices = [i for i, prob in enumerate(probs) if prob > MIN_CONFIDENCE]
        boxes = boxes[confident_indices]

        # if no boxes remain after filtering, treat as no detection
        if len(boxes) == 0:
            return None, []

        # extract aligned face tensors for the confident detections
        aligned_faces = mtcnn.extract(img_rgb, boxes, save_path=None)

        with torch.no_grad():
            embeddings = resnet(aligned_faces.to(DEVICE)).cpu().numpy()

        known_encodings_np = np.array(known_face_encodings)
        for embedding in embeddings:
            # use cosine similarity
            # normalize both the known encodings and the current embedding
            norm_known = np.linalg.norm(known_encodings_np, axis=1, keepdims=True)
            normalized_known = known_encodings_np / norm_known

            norm_embedding = np.linalg.norm(embedding)
            normalized_embedding = embedding / norm_embedding

            # compute dot product (that is: cosine similarity for normalized vectors)
            similarities = np.dot(normalized_known, normalized_embedding.T)

            max_similarity = np.max(similarities)
            name = "Unknown"

            if max_similarity > COSINE_THRESHOLD:
                best_match_index = np.argmax(similarities)
                name = known_face_names[best_match_index]

            # append confidence to name
            names.append(f"{name} ({max_similarity:.2f})")

    return boxes, names


def main(args):
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(
            f"[INFO] Created directory: {KNOWN_FACES_DIR}. Please add images of known faces here."
        )
        return

    # check for re-encoding flag
    if args.re_encode or not os.path.exists(ENCODINGS_FILE):
        print(
            "[INFO] Re-encoding flag set or no encoding file found. Encoding known faces..."
        )
        encode_known_faces(KNOWN_FACES_DIR, ENCODINGS_FILE)

    known_face_encodings, known_face_names = load_encodings(ENCODINGS_FILE)

    if not known_face_names:
        print(
            f"[ERROR] No faces were encoded. Add images to '{KNOWN_FACES_DIR}' and run with '--re-encode'."
        )
        return

    print(f"[INFO] Loaded {len(known_face_encodings)} known face encodings.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not found or could not be opened.")
        return

    print("[INFO] Starting face recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        boxes, names = recognize_faces(frame, known_face_encodings, known_face_names)

        if boxes is not None:
            for box, name in zip(boxes, names):
                box = [int(b) for b in box]
                cv2.rectangle(
                    frame, (box[0], box[1]), (box[2], box[3]), (0, 165, 255), 2
                )
                label_size, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(
                    frame,
                    (box[0], box[1] - label_size[1] - 10),
                    (box[0] + label_size[0], box[1] - 10),
                    (0, 165, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    name,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


if __name__ == "__main__":
    # set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Face Recognition with FaceNet-PyTorch"
    )
    parser.add_argument(
        '--re-encode', action='store_true', help='Force re-encoding of known faces.'
    )
    args = parser.parse_args()
    main(args)
