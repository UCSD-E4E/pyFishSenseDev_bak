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
        # Create a private argument parser object. The program is stored as the app name, pulled from
        # The init file. The description is the description

        self._parser = argparse.ArgumentParser(
            prog=__app_name__,
            description="Engineers for Exploration tool for calibrating cameras.",
        )

        # Add the version argument. If this goes in, then the version of the current program
        # can be outputted later.

        self._parser.add_argument(
            "-v",
            "--version",
            help="Displays the version number of the Cli.",
            action="store_true",
        )

        # There are multiple different use cases for this program. Thus, adding subparsers
        # is useful for better functionality.

        self._subparsers = self._parser.add_subparsers(dest="command")

        # These are all the subparsers for this program. Each _add_subparser call takes in a an
        # argparser. These are all the different command line argument formats that this program
        # can take in.

        # So, each of these classes below inherit from the CliCommand absract class.

        # This class comes from extract_calibration_images.py. Can ignore for Olympus
        self._add_subparser(ExtractCalibrationImagesCommand())

        # This class comes from calibrate_command.py. THIS needs to be looked into more.
        # Should probably create some sort of similar class, like CalibrateCommandMono
        # Or something. Later problem
        self._add_subparser(CalibrateCommand())

        # This is only for stereo. Can ignore for olympus.
        self._add_subparser(TuneDisparityCommand())

        # This will stay as is. Unnecessary addition for Olympus.
        self._add_subparser(VersionCommand())

    # Method for adding subparsers.

    def _add_subparser(self, command: CliCommand):
        subparser = self._subparsers.add_parser(command.name, help=command.help)
        command.parser = subparser
        subparser.set_defaults(command=command)

    def execute(self):
        args = self._parser.parse_args()

        if args.version:
            print(__version__)

        # If we're looking for a command line argument, go here. This is what is relevant.
        elif args.command:
            # Step into the builder. Builder is a method in the given command Class.
            # In our case, it's CalibrateCommand
            parsed_args = args.command.builder.parse_args(args)
            exit_code = args.command.execute(args, parsed_args) or 0

            if exit_code != 0:
                exit(exit_code)


Cli().execute()
