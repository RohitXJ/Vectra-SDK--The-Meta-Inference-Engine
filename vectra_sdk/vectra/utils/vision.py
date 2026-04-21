import cv2
import time

class LiveStreamInference:
    """
    Utility class for real-time inference on camera feeds.
    """
    def __init__(self, classifier):
        self.classifier = classifier
        
    def start(self, camera_index=0, window_name="Vectra SDK Live Feed"):
        """
        Starts the camera stream and displays predictions on-screen.
        Press 'q' to exit.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return

        print(f"Starting live feed. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run inference
            try:
                result = self.classifier.predict(frame)
                label = result['label']
                conf = result.get('confidence', 0.0)
                
                # Overlay text
                text = f"{label} ({conf:.2f})"
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                
                cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, color, 2, cv2.LINE_AA)
            except Exception as e:
                cv2.putText(frame, f"Error: {str(e)}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(window_name, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
