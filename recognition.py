import os
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
import pickle

# --- configuration ---
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings_facenet.pkl"
TOLERANCE = 0.8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- models ---
 # ensure "keep_all=True" for multi-face
mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device=DEVICE) 
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)



def encode_known_faces(known_faces_dir, encodings_file):
    known_face_encodings = []
    known_face_names = []

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

        # Detect and align face
        face = mtcnn(img_rgb)
        if face is None:
            print(f"[WARNING] No face detected in {filename}. Skipping.")
            continue

        with torch.no_grad():
            embedding = resnet(face.unsqueeze(0).to(DEVICE)).cpu().numpy()[0]

        known_face_encodings.append(embedding)
        known_face_names.append(name)
        print(f"[INFO] Encoded: {name}")

    with open(encodings_file, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)


def load_encodings(encodings_file):
    if not os.path.exists(encodings_file):
        return [], []
    with open(encodings_file, 'rb') as f:
        return pickle.load(f)


def recognize_faces(frame, known_face_encodings, known_face_names):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(img_rgb)

    names = []
    aligned_faces = mtcnn.extract(img_rgb, boxes, save_path=None) if boxes is not None else None

    if aligned_faces is not None:
        with torch.no_grad():
            embeddings = resnet(aligned_faces.to(DEVICE)).cpu().numpy()

        for embedding in embeddings:
            distances = np.linalg.norm(np.array(known_face_encodings) - embedding, axis=1)
            if len(distances) == 0 or np.min(distances) > TOLERANCE:
                name = "Unknown"
            else:
                name = known_face_names[np.argmin(distances)]
            names.append(name)
    return boxes, names


def main():
    if not os.path.exists(ENCODINGS_FILE):
        print("[INFO] Encoding known faces...")
        encode_known_faces(KNOWN_FACES_DIR, ENCODINGS_FILE)

    known_face_encodings, known_face_names = load_encodings(ENCODINGS_FILE)
    print(f"[INFO] Loaded {len(known_face_encodings)} known faces.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not found.")
        return

    print("[INFO] Starting face recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, names = recognize_faces(frame, known_face_encodings, known_face_names)

        if boxes is not None:
            for box, name in zip(boxes, names):
                box = [int(b) for b in box]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


if __name__ == "__main__":
    main()
