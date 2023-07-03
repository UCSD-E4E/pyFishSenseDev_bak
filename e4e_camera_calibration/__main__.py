import argparse

from __init__ import __app_name__, __version__
from e4e_camera_calibration.commands.cli_command import CliCommand
from e4e_camera_calibration.commands.calibrate_command import CalibrateCommand
from e4e_camera_calibration.commands.extract_calibration_images import (
    ExtractCalibrationImagesCommand,
)
from e4e_camera_calibration.commands.tune_disparity_command import TuneDisparityCommand
from e4e_camera_calibration.commands.version_command import VersionCommand


class Cli:
    def __init__(self) -> None:
        self._parser = argparse.ArgumentParser(
            prog=__app_name__,
            description="Engineers for Exploration tool for calibrating cameras.",
        )

        self._parser.add_argument(
            "-v",
            "--version",
            help="Displays the version number of the Cli.",
            action="store_true",
        )

        self._subparsers = self._parser.add_subparsers(dest="command")
        self._add_subparser(ExtractCalibrationImagesCommand())
        self._add_subparser(CalibrateCommand())
        self._add_subparser(TuneDisparityCommand())
        self._add_subparser(VersionCommand())

    def _add_subparser(self, command: CliCommand):
        subparser = self._subparsers.add_parser(command.name, help=command.help)
        command.parser = subparser
        subparser.set_defaults(command=command)

    def execute(self):
        args = self._parser.parse_args()

        if args.version:
            print(__version__)
        elif args.command:
            parsed_args = args.command.builder.parse_args(args)
            exit_code = args.command.execute(args, parsed_args) or 0

            if exit_code != 0:
                exit(exit_code)


Cli().execute()
