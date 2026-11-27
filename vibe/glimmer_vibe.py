"""Local phrase training and text generation for Muse Glimmer 30B."""

from particle_sliders_vibe.text_cli import main as text_main


def main():
    return text_main(model="glimmer")


if __name__ == "__main__":
    main()
