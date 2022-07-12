from detector import ObjectDetection

# usage: python main.py

if __name__ == "__main__":
    
    video_url = "https://www.youtube.com/watch?v=wqctLW0Hb_0"
    
    # Create a new object and execute.
    object_detection = ObjectDetection()
    object_detection.detect(video_url=video_url)