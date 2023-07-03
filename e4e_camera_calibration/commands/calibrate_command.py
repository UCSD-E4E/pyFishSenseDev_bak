from argparse import ArgumentParser, Namespace
from e4e_camera_calibration.commands.cli_command import CliCommand
from e4e_camera_calibration.cameras.calibrated_camera_base import CalibratedCameraBase
from e4e_camera_calibration.cameras.calibrated_stereo_camera import CalibratedStereoCamera
from argument_parser_builder import ArgumentParserBuilder, ParsedArguments


class CalibrateCommand(CliCommand):
    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "calibrate"

    @property
    def help(self) -> str:
        return "Generates a camera calibration matrix based off the camera folder."

    def execute(self, args: Namespace, parsed_arguments: ParsedArguments):
        parsed_arguments.camera.load(args.calibration_directory)

        calibrated_camera: CalibratedCameraBase = None
        if parsed_arguments.camera.number_of_sensors == 2:
            calibrated_camera = CalibratedStereoCamera(parsed_arguments.camera)
        else:
            raise ValueError(
                f"Camera has {parsed_arguments.camera.number_of_sensors} sensors.  This is not currently supported."
            )

        calibrated_camera.calibrate(
            display_calibration_error=args.display_error, max_error=args.max_error
        )

        calibrated_camera.save_calibration(args.output)

    def _set_parser(self, parser: ArgumentParser, builder: ArgumentParserBuilder):
        builder.add_camera_parameters().add_calibration_directory_parameter().add_display_error_parameter()

        parser.add_argument(
            "-m",
            "--max-error",
            type=int,
            default=0.7,
            help="The maximum calibration error allowed.",
        )

        parser.add_argument(
            "-o",
            "--output",
            required=True,
            help="Where to save the results of the calibration.",
        )
