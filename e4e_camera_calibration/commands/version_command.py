from argparse import Namespace
from argument_parser_builder import ParsedArguments
from e4e_camera_calibration.commands.cli_command import CliCommand

from __init__ import __version__


class VersionCommand(CliCommand):
    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "version"

    @property
    def help(self) -> str:
        return "Displays the version number of the Cli."

    def execute(self, args: Namespace, parsed_arguments: ParsedArguments):
        print(__version__)
