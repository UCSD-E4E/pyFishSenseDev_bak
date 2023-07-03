from argparse import ArgumentParser, Namespace

from e4e_camera_calibration.cameras.factory import CAMERA_MAP, str2camera
from e4e_camera_calibration.cameras.camera import Camera


class ParsedArguments:
    def __init__(self, camera: Camera) -> None:
        self.camera = camera


class ArgumentParserBuilder:
    def __init__(self, parser: ArgumentParser) -> None:
        self._parser = parser
        self.camera: Camera

    @property
    def parser(self):
        return self._parser

    def add_camera_name_parameter(self):
        self.parser.add_argument(
            "-n",
            "--camera-name",
            type=str,
            default=None,
            help="The name of the name of the camera",
        )

        return self

    def add_calibration_directory_parameter(self):
        self.parser.add_argument(
            "-i",
            "--calibration-directory",
            required=True,
            help="The path to a list of calibration images.",
        )

        return self

    def add_camera_parameters(self):
        return (
            self.add_camera_name_parameter()
            .add_camera_parameter()
            .add_serial_number_parameter()
        )

    def add_camera_parameter(self):
        self.parser.add_argument(
            "-c",
            "--camera",
            required=True,
            choices=CAMERA_MAP.keys(),
            help="The model of the camera to calibrate.",
        )

        return self

    def add_display_error_parameter(self):
        self.parser.add_argument(
            "-e",
            "--display-error",
            help="Display the calibration error.",
            action="store_true",
        )

        return self

    def add_serial_number_parameter(self):
        self.parser.add_argument(
            "-s",
            "--serial-number",
            type=str,
            default=None,
            help="The serial number of the camera being calibrated.",
        )

        return self

    def parse_args(self, args: Namespace):
        camera = None
        if hasattr(args, "camera"):
            camera = str2camera(
                args.camera, name=args.camera_name, serial_number=args.serial_number
            )

        return ParsedArguments(camera)
