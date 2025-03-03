import modal

# Create an app instance
app = modal.App("my-video-processing-app")

# Build the image from the Dockerfile
image = modal.Image.from_dockerfile("Dockerfile")

@app.function(image=image)
def process_video():
    print("Processing video using the unified image")
