from src import recorder

def main():
    cap = recorder.initialize_camera()
    out = recorder.initialize_video_writer()
    recorder.record_video(cap, out, duration=10)
    recorder.release_resources(cap, out)

if __name__ == "__main__":
    main()