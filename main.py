from src.load_config import load_yaml_config
from src.preprocess_data import get_data
from src.controller import get_controller_positions
from src.find_centroids import get_centroids
from src.matching_blobs_led import match_blobs_to_leds


def main():
    config = load_yaml_config('./config/config.yml')

    controller_3d_pos = get_controller_positions(config["controllers"])

    dataloader = get_data(config["data"])

    # Iterate
    for image_batch in dataloader:
        image = image_batch[0]
        blob_centroids = get_centroids(image, config["blob_detection"])
        match_blobs_to_leds(blob_centroids, controller_3d_pos)

        print(1)


if __name__ == '__main__':
    main()
