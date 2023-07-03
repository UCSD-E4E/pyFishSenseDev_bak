from argparse import ArgumentParser, Namespace
from argument_parser_builder import ArgumentParserBuilder, ParsedArguments

from e4e_camera_calibration.commands.cli_command import CliCommand
from e4e_camera_calibration.cameras.calibrated_stereo_camera import (
    CalibratedStereoCamera,
)
from e4e_camera_calibration.disparity.factory import str2disparity, DISPARITY_MAP


class TuneDisparityCommand(CliCommand):
    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "tune-disparity"

    @property
    def help(self) -> str:
        return "Tunes the parameters for disparity using Sherlock."

    def execute(self, args: Namespace, parsed_arguments: ParsedArguments):
        calibrated_camera = CalibratedStereoCamera(parsed_arguments.camera)
        calibrated_camera.load_calibration(args.calibration_tables)

        disparity = str2disparity(args.algorithm, calibrated_camera)
        disparity.calibrate(
            args.calibration_directory, display_calibration_error=args.display_error
        )

    def _set_parser(self, parser: ArgumentParser, builder: ArgumentParserBuilder):
        builder.add_camera_parameters().add_calibration_directory_parameter().add_display_error_parameter()

        parser.add_argument(
            "-t",
            "--calibration-tables",
            type=str,
            required=True,
            help="The calibration tables generated from the calibrate command.",
        )

        parser.add_argument(
            "-a",
            "--algorithm",
            type=str,
            default="SGBM",
            choices=DISPARITY_MAP.keys(),
            help="The block matching algorithm used for generating the disparity map.",
        )
